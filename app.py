import os
import sys
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ['GRADIO_ANALYTICS_ENABLED'] = '0'
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), 'sd-scripts'))
import subprocess
import gradio as gr
from PIL import Image
import torch
import uuid
import shutil
import json
import yaml
from slugify import slugify
from transformers import AutoProcessor, AutoModelForCausalLM
from gradio_logsview import LogsView, LogsViewRunner
from huggingface_hub import hf_hub_download, HfApi
from library import flux_train_utils, huggingface_util
from argparse import Namespace
import train_network
import toml
import re
MAX_IMAGES = 150

with open('models.yaml', 'r') as file:
    models = yaml.safe_load(file)

def readme(base_model, lora_name, instance_prompt, sample_prompts):

    # 模型许可证
    model_config = models[base_model]
    model_file = model_config["file"]
    base_model_name = model_config["base"]
    license = None
    license_name = None
    license_link = None
    license_items = []
    if "license" in model_config:
        license = model_config["license"]
        license_items.append(f"license: {license}")
    if "license_name" in model_config:
        license_name = model_config["license_name"]
        license_items.append(f"license_name: {license_name}")
    if "license_link" in model_config:
        license_link = model_config["license_link"]
        license_items.append(f"license_link: {license_link}")
    license_str = "\n".join(license_items)
    print(f"license_items={license_items}")
    print(f"license_str = {license_str}")

    # 标签
    tags = [ "text-to-image", "flux", "lora", "diffusers", "template:sd-lora", "fluxgym" ]

    # 小部件
    widgets = []
    sample_image_paths = []
    output_name = slugify(lora_name)
    samples_dir = resolve_path_without_quotes(f"outputs/{output_name}/sample")
    try:
        for filename in os.listdir(samples_dir):
            # 文件名格式：[name]_[steps]_[index]_[timestamp].png
            match = re.search(r"_(\d+)_(\d+)_(\d+)\.png$", filename)
            if match:
                steps, index, timestamp = int(match.group(1)), int(match.group(2)), int(match.group(3))
                sample_image_paths.append((steps, index, f"sample/{filename}"))

        # 按数字索引排序
        sample_image_paths.sort(key=lambda x: x[0], reverse=True)

        final_sample_image_paths = sample_image_paths[:len(sample_prompts)]
        final_sample_image_paths.sort(key=lambda x: x[1])
        for i, prompt in enumerate(sample_prompts):
            _, _, image_path = final_sample_image_paths[i]
            widgets.append(
                {
                    "text": prompt,
                    "output": {
                        "url": image_path
                    },
                }
            )
    except:
        print(f"没有样本")
    dtype = "torch.bfloat16"
    # 构建README内容
    readme_content = f"""---
tags:
{yaml.dump(tags, indent=4).strip()}
{"widget:" if os.path.isdir(samples_dir) else ""}
{yaml.dump(widgets, indent=4).strip() if widgets else ""}
base_model: {base_model_name}
{"instance_prompt: " + instance_prompt if instance_prompt else ""}
{license_str}
---

# {lora_name}

一个在本地计算机上使用[Fluxgym](https://github.com/cocktailpeanut/fluxgym)训练的Flux LoRA

<Gallery />

## 触发词

{"你应该使用`" + instance_prompt + "`来触发图像生成。" if instance_prompt else "没有定义触发词。"}

## 下载模型并在ComfyUI、AUTOMATIC1111、SD.Next、Invoke AI、Forge等中使用

该模型的权重以Safetensors格式提供。

"""
    return readme_content

def account_hf():
    try:
        with open("HF_TOKEN", "r") as file:
            token = file.read()
            api = HfApi(token=token)
            try:
                account = api.whoami()
                return { "token": token, "account": account['name'] }
            except:
                return None
    except:
        return None

"""
hf_logout.click(fn=logout_hf, outputs=[hf_token, hf_login, hf_logout, repo_owner])
"""
def logout_hf():
    os.remove("HF_TOKEN")
    global current_account
    current_account = account_hf()
    print(f"current_account={current_account}")
    return gr.update(value=""), gr.update(visible=True), gr.update(visible=False), gr.update(value="", visible=False)


"""
hf_login.click(fn=login_hf, inputs=[hf_token], outputs=[hf_token, hf_login, hf_logout, repo_owner])
"""
def login_hf(hf_token):
    api = HfApi(token=hf_token)
    try:
        account = api.whoami()
        if account != None:
            if "name" in account:
                with open("HF_TOKEN", "w") as file:
                    file.write(hf_token)
                global current_account
                current_account = account_hf()
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(value=current_account["account"], visible=True)
        return gr.update(), gr.update(), gr.update(), gr.update()
    except:
        print(f"错误的hf_token")
        return gr.update(), gr.update(), gr.update(), gr.update()

def upload_hf(base_model, lora_rows, repo_owner, repo_name, repo_visibility, hf_token):
    src = lora_rows
    repo_id = f"{repo_owner}/{repo_name}"
    gr.Info(f"正在上传到Huggingface，请稍候...", duration=None)
    args = Namespace(
        huggingface_repo_id=repo_id,
        huggingface_repo_type="model",
        huggingface_repo_visibility=repo_visibility,
        huggingface_path_in_repo="",
        huggingface_token=hf_token,
        async_upload=False
    )
    print(f"upload_hf args={args}")
    huggingface_util.upload(args=args, src=src)
    gr.Info(f"[上传完成] https://huggingface.co/{repo_id}", duration=None)

def load_captioning(uploaded_files, concept_sentence):
    uploaded_images = [file for file in uploaded_files if not file.endswith('.txt')]
    txt_files = [file for file in uploaded_files if file.endswith('.txt')]
    txt_files_dict = {os.path.splitext(os.path.basename(txt_file))[0]: txt_file for txt_file in txt_files}
    updates = []
    if len(uploaded_images) <= 1:
        raise gr.Error(
            "请上传至少2张图片来训练你的模型（默认设置下理想数量为4-30张）"
        )
    elif len(uploaded_images) > MAX_IMAGES:
        raise gr.Error(f"目前只允许最多{MAX_IMAGES}张图片进行训练")
    # 更新captioning_area
    # for _ in range(3):
    updates.append(gr.update(visible=True))
    # 更新每个captioning行和图像的可见性和图像
    for i in range(1, MAX_IMAGES + 1):
        # 确定当前行和图像是否应可见
        visible = i <= len(uploaded_images)

        # 更新captioning行的可见性
        updates.append(gr.update(visible=visible))

        # 更新图像组件 - 如果可用则显示图像，否则隐藏
        image_value = uploaded_images[i - 1] if visible else None
        updates.append(gr.update(value=image_value, visible=visible))

        corresponding_caption = False
        if(image_value):
            base_name = os.path.splitext(os.path.basename(image_value))[0]
            if base_name in txt_files_dict:
                with open(txt_files_dict[base_name], 'r') as file:
                    corresponding_caption = file.read()

        # 更新captioning区域的值
        text_value = corresponding_caption if visible and corresponding_caption else concept_sentence if visible and concept_sentence else None
        updates.append(gr.update(value=text_value, visible=visible))

    # 更新样本caption区域
    updates.append(gr.update(visible=True))
    updates.append(gr.update(visible=True))

    return updates

def hide_captioning():
    return gr.update(visible=False), gr.update(visible=False)

def resize_image(image_path, output_path, size):
    with Image.open(image_path) as img:
        width, height = img.size
        if width < height:
            new_width = size
            new_height = int((size/width) * height)
        else:
            new_height = size
            new_width = int((size/height) * width)
        print(f"调整大小 {image_path} : {new_width}x{new_height}")
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_resized.save(output_path)

def create_dataset(destination_folder, size, *inputs):
    print("创建数据集")
    images = inputs[0]
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for index, image in enumerate(images):
        # 将图像复制到数据集文件夹
        new_image_path = shutil.copy(image, destination_folder)

        # 如果是caption文本文件，跳过下一步
        ext = os.path.splitext(new_image_path)[-1].lower()
        if ext == '.txt':
            continue

        # 调整图像大小
        resize_image(new_image_path, new_image_path, size)

        # 复制captions

        original_caption = inputs[index + 1]

        image_file_name = os.path.basename(new_image_path)
        caption_file_name = os.path.splitext(image_file_name)[0] + ".txt"
        caption_path = resolve_path_without_quotes(os.path.join(destination_folder, caption_file_name))
        print(f"image_path={new_image_path}, caption_path = {caption_path}, original_caption={original_caption}")
        # 如果caption_path存在，不写入
        if os.path.exists(caption_path):
            print(f"{caption_path} 已存在，使用现有的.txt文件")
        else:
            print(f"{caption_path} 创建一个.txt caption文件")
            with open(caption_path, 'w') as file:
                file.write(original_caption)

    print(f"destination_folder {destination_folder}")
    return destination_folder


def run_captioning(images, concept_sentence, *captions):
    print(f"运行captioning")
    print(f"concept sentence {concept_sentence}")
    print(f"captions {captions}")
    # 加载内部以不消耗训练资源
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    torch_dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        "multimodalart/Florence-2-large-no-flash-attn", torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained("multimodalart/Florence-2-large-no-flash-attn", trust_remote_code=True)

    captions = list(captions)
    for i, image_path in enumerate(images):
        print(captions[i])
        if isinstance(image_path, str):  # 如果图像是文件路径
            image = Image.open(image_path).convert("RGB")

        prompt = "<DETAILED_CAPTION>"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
        print(f"inputs {inputs}")

        generated_ids = model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3
        )
        print(f"generated_ids {generated_ids}")

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        print(f"generated_text: {generated_text}")
        parsed_answer = processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )
        print(f"parsed_answer = {parsed_answer}")
        caption_text = parsed_answer["<DETAILED_CAPTION>"].replace("The image shows ", "")
        print(f"caption_text = {caption_text}, concept_sentence={concept_sentence}")
        if concept_sentence:
            caption_text = f"{concept_sentence} {caption_text}"
        captions[i] = caption_text

        yield captions
    model.to("cpu")
    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and v:
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def download(base_model):
    model = models[base_model]
    model_file = model["file"]
    repo = model["repo"]

    # 下载unet
    if base_model == "flux-dev" or base_model == "flux-schnell":
        unet_folder = "models/unet"
    else:
        unet_folder = f"models/unet/{repo}"
    unet_path = os.path.join(unet_folder, model_file)
    if not os.path.exists(unet_path):
        os.makedirs(unet_folder, exist_ok=True)
        gr.Info(f"正在下载基础模型: {base_model}，请稍候。（您可以检查终端以查看下载进度）", duration=None)
        print(f"下载 {base_model}")
        hf_hub_download(repo_id=repo, local_dir=unet_folder, filename=model_file)

    # 下载vae
    vae_folder = "models/vae"
    vae_path = os.path.join(vae_folder, "ae.sft")
    if not os.path.exists(vae_path):
        os.makedirs(vae_folder, exist_ok=True)
        gr.Info(f"正在下载vae")
        print(f"下载ae.sft...")
        hf_hub_download(repo_id="cocktailpeanut/xulf-dev", local_dir=vae_folder, filename="ae.sft")

    # 下载clip
    clip_folder = "models/clip"
    clip_l_path = os.path.join(clip_folder, "clip_l.safetensors")
    if not os.path.exists(clip_l_path):
        os.makedirs(clip_folder, exist_ok=True)
        gr.Info(f"正在下载clip...")
        print(f"下载clip_l.safetensors")
        hf_hub_download(repo_id="comfyanonymous/flux_text_encoders", local_dir=clip_folder, filename="clip_l.safetensors")

    # 下载t5xxl
    t5xxl_path = os.path.join(clip_folder, "t5xxl_fp16.safetensors")
    if not os.path.exists(t5xxl_path):
        print(f"下载t5xxl_fp16.safetensors")
        gr.Info(f"正在下载t5xxl...")
        hf_hub_download(repo_id="comfyanonymous/flux_text_encoders", local_dir=clip_folder, filename="t5xxl_fp16.safetensors")


def resolve_path(p):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    norm_path = os.path.normpath(os.path.join(current_dir, p))
    return f"\"{norm_path}\""
def resolve_path_without_quotes(p):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    norm_path = os.path.normpath(os.path.join(current_dir, p))
    return norm_path

def gen_sh(
    base_model,
    output_name,
    resolution,
    seed,
    workers,
    learning_rate,
    network_dim,
    max_train_epochs,
    save_every_n_epochs,
    timestep_sampling,
    guidance_scale,
    vram,
    sample_prompts,
    sample_every_n_steps,
    *advanced_components
):

    print(f"gen_sh: network_dim:{network_dim}, max_train_epochs={max_train_epochs}, save_every_n_epochs={save_every_n_epochs}, timestep_sampling={timestep_sampling}, guidance_scale={guidance_scale}, vram={vram}, sample_prompts={sample_prompts}, sample_every_n_steps={sample_every_n_steps}")

    output_dir = resolve_path(f"outputs/{output_name}")
    sample_prompts_path = resolve_path(f"outputs/{output_name}/sample_prompts.txt")

    line_break = "\\"
    file_type = "sh"
    if sys.platform == "win32":
        line_break = "^"
        file_type = "bat"

    ############# Sample args ########################
    sample = ""
    if len(sample_prompts) > 0 and sample_every_n_steps > 0:
        sample = f"""--sample_prompts={sample_prompts_path} --sample_every_n_steps="{sample_every_n_steps}" {line_break}"""


    ############# Optimizer args ########################
#    if vram == "8G":
#        optimizer = f"""--optimizer_type adafactor {line_break}
#    --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" {line_break}
#        --split_mode {line_break}
#        --network_args "train_blocks=single" {line_break}
#        --lr_scheduler constant_with_warmup {line_break}
#        --max_grad_norm 0.0 {line_break}"""
    if vram == "16G":
        # 16G VRAM
        optimizer = f"""--optimizer_type adafactor {line_break}
  --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" {line_break}
  --lr_scheduler constant_with_warmup {line_break}
  --max_grad_norm 0.0 {line_break}"""
    elif vram == "12G":
      # 12G VRAM
        optimizer = f"""--optimizer_type adafactor {line_break}
  --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" {line_break}
  --split_mode {line_break}
  --network_args "train_blocks=single" {line_break}
  --lr_scheduler constant_with_warmup {line_break}
  --max_grad_norm 0.0 {line_break}"""
    else:
        # 20G+ VRAM
        optimizer = f"--optimizer_type adamw8bit {line_break}"


    #######################################################
    model_config = models[base_model]
    model_file = model_config["file"]
    repo = model_config["repo"]
    if base_model == "flux-dev" or base_model == "flux-schnell":
        model_folder = "models/unet"
    else:
        model_folder = f"models/unet/{repo}"
    model_path = os.path.join(model_folder, model_file)
    pretrained_model_path = resolve_path(model_path)

    clip_path = resolve_path("models/clip/clip_l.safetensors")
    t5_path = resolve_path("models/clip/t5xxl_fp16.safetensors")
    ae_path = resolve_path("models/vae/ae.sft")
    sh = f"""accelerate launch {line_break}
  --mixed_precision bf16 {line_break}
  --num_cpu_threads_per_process 1 {line_break}
  sd-scripts/flux_train_network.py {line_break}
  --pretrained_model_name_or_path {pretrained_model_path} {line_break}
  --clip_l {clip_path} {line_break}
  --t5xxl {t5_path} {line_break}
  --ae {ae_path} {line_break}
  --cache_latents_to_disk {line_break}
  --save_model_as safetensors {line_break}
  --sdpa --persistent_data_loader_workers {line_break}
  --max_data_loader_n_workers {workers} {line_break}
  --seed {seed} {line_break}
  --gradient_checkpointing {line_break}
  --mixed_precision bf16 {line_break}
  --save_precision bf16 {line_break}
  --network_module networks.lora_flux {line_break}
  --network_dim {network_dim} {line_break}
  {optimizer}{sample}
  --learning_rate {learning_rate} {line_break}
  --cache_text_encoder_outputs {line_break}
  --cache_text_encoder_outputs_to_disk {line_break}
  --fp8_base {line_break}
  --highvram {line_break}
  --max_train_epochs {max_train_epochs} {line_break}
  --save_every_n_epochs {save_every_n_epochs} {line_break}
  --dataset_config {resolve_path(f"outputs/{output_name}/dataset.toml")} {line_break}
  --output_dir {output_dir} {line_break}
  --output_name {output_name} {line_break}
  --timestep_sampling {timestep_sampling} {line_break}
  --discrete_flow_shift 3.1582 {line_break}
  --model_prediction_type raw {line_break}
  --guidance_scale {guidance_scale} {line_break}
  --loss_type l2 {line_break}"""
   


    ############# Advanced args ########################
    global advanced_component_ids
    global original_advanced_component_values
   
    # 检查脏
    print(f"original_advanced_component_values = {original_advanced_component_values}")
    advanced_flags = []
    for i, current_value in enumerate(advanced_components):
#        print(f"compare {advanced_component_ids[i]}: old={original_advanced_component_values[i]}, new={current_value}")
        if original_advanced_component_values[i] != current_value:
            # 脏
            if current_value == True:
                # Boolean
                advanced_flags.append(advanced_component_ids[i])
            else:
                # string
                advanced_flags.append(f"{advanced_component_ids[i]} {current_value}")

    if len(advanced_flags) > 0:
        advanced_flags_str = f" {line_break}\n  ".join(advanced_flags)
        sh = sh + "\n  " + advanced_flags_str

    return sh

def gen_toml(
  dataset_folder,
  resolution,
  class_tokens,
  num_repeats
):
    toml = f"""[general]
shuffle_caption = false
caption_extension = '.txt'
keep_tokens = 1

[[datasets]]
resolution = {resolution}
batch_size = 1
keep_tokens = 1

  [[datasets.subsets]]
  image_dir = '{resolve_path_without_quotes(dataset_folder)}'
  class_tokens = '{class_tokens}'
  num_repeats = {num_repeats}"""
    return toml

def update_total_steps(max_train_epochs, num_repeats, images):
    try:
        num_images = len(images)
        total_steps = max_train_epochs * num_images * num_repeats
        print(f"max_train_epochs={max_train_epochs} num_images={num_images}, num_repeats={num_repeats}, total_steps={total_steps}")
        return gr.update(value = total_steps)
    except:
        print("")

def set_repo(lora_rows):
    selected_name = os.path.basename(lora_rows)
    return gr.update(value=selected_name)

def get_loras():
    try:
        outputs_path = resolve_path_without_quotes(f"outputs")
        files = os.listdir(outputs_path)
        folders = [os.path.join(outputs_path, item) for item in files if os.path.isdir(os.path.join(outputs_path, item)) and item != "sample"]
        folders.sort(key=lambda file: os.path.getctime(file), reverse=True)
        return folders
    except Exception as e:
        return []

def get_samples(lora_name):
    output_name = slugify(lora_name)
    try:
        samples_path = resolve_path_without_quotes(f"outputs/{output_name}/sample")
        files = [os.path.join(samples_path, file) for file in os.listdir(samples_path)]
        files.sort(key=lambda file: os.path.getctime(file), reverse=True)
        return files
    except:
        return []

def start_training(
    base_model,
    lora_name,
    train_script,
    train_config,
    sample_prompts,
):
    # 写入自定义脚本和toml
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)
    if not os.path.exists("outputs"):
        os.makedirs("outputs", exist_ok=True)
    output_name = slugify(lora_name)
    output_dir = resolve_path_without_quotes(f"outputs/{output_name}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    download(base_model)

    file_type = "sh"
    if sys.platform == "win32":
        file_type = "bat"

    sh_filename = f"train.{file_type}"
    sh_filepath = resolve_path_without_quotes(f"outputs/{output_name}/{sh_filename}")
    with open(sh_filepath, 'w', encoding="utf-8") as file:
        file.write(train_script)
    gr.Info(f"生成的训练脚本在 {sh_filename}")


    dataset_path = resolve_path_without_quotes(f"outputs/{output_name}/dataset.toml")
    with open(dataset_path, 'w', encoding="utf-8") as file:
        file.write(train_config)
    gr.Info(f"生成的dataset.toml")

    sample_prompts_path = resolve_path_without_quotes(f"outputs/{output_name}/sample_prompts.txt")
    with open(sample_prompts_path, 'w', encoding='utf-8') as file:
        file.write(sample_prompts)
    gr.Info(f"生成的sample_prompts.txt")

    # 训练
    if sys.platform == "win32":
        command = sh_filepath
    else:
        command = f"bash \"{sh_filepath}\""

    # 使用Popen运行命令并实时捕获输出
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['LOG_LEVEL'] = 'DEBUG'
    runner = LogsViewRunner()
    cwd = os.path.dirname(os.path.abspath(__file__))
    gr.Info(f"开始训练")
    yield from runner.run_command([command], cwd=cwd)
    yield runner.log(f"Runner: {runner}")

    # 生成README
    config = toml.loads(train_config)
    concept_sentence = config['datasets'][0]['subsets'][0]['class_tokens']
    print(f"concept_sentence={concept_sentence}")
    print(f"lora_name {lora_name}, concept_sentence={concept_sentence}, output_name={output_name}")
    sample_prompts_path = resolve_path_without_quotes(f"outputs/{output_name}/sample_prompts.txt")
    with open(sample_prompts_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    sample_prompts = [line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"]
    md = readme(base_model, lora_name, concept_sentence, sample_prompts)
    readme_path = resolve_path_without_quotes(f"outputs/{output_name}/README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(md)

    gr.Info(f"训练完成。请检查outputs文件夹以获取LoRA文件。", duration=None)


def update(
    base_model,
    lora_name,
    resolution,
    seed,
    workers,
    class_tokens,
    learning_rate,
    network_dim,
    max_train_epochs,
    save_every_n_epochs,
    timestep_sampling,
    guidance_scale,
    vram,
    num_repeats,
    sample_prompts,
    sample_every_n_steps,
    *advanced_components,
):
    output_name = slugify(lora_name)
    dataset_folder = str(f"datasets/{output_name}")
    sh = gen_sh(
        base_model,
        output_name,
        resolution,
        seed,
        workers,
        learning_rate,
        network_dim,
        max_train_epochs,
        save_every_n_epochs,
        timestep_sampling,
        guidance_scale,
        vram,
        sample_prompts,
        sample_every_n_steps,
        *advanced_components,
    )
    toml = gen_toml(
        dataset_folder,
        resolution,
        class_tokens,
        num_repeats
    )
    return gr.update(value=sh), gr.update(value=toml), dataset_folder

"""
demo.load(fn=loaded, js=js, outputs=[hf_token, hf_login, hf_logout, hf_account])
"""
def loaded():
    global current_account
    current_account = account_hf()
    print(f"current_account={current_account}")
    if current_account != None:
        return gr.update(value=current_account["token"]), gr.update(visible=False), gr.update(visible=True), gr.update(value=current_account["account"], visible=True)
    else:
        return gr.update(value=""), gr.update(visible=True), gr.update(visible=False), gr.update(value="", visible=False)

def update_sample(concept_sentence):
    return gr.update(value=concept_sentence)

def refresh_publish_tab():
    loras = get_loras()
    return gr.Dropdown(label="已训练的LoRAs", choices=loras)

def init_advanced():
    # if basic_args
    basic_args = {
        'pretrained_model_name_or_path',
        'clip_l',
        't5xxl',
        'ae',
        'cache_latents_to_disk',
        'save_model_as',
        'sdpa',
        'persistent_data_loader_workers',
        'max_data_loader_n_workers',
        'seed',
        'gradient_checkpointing',
        'mixed_precision',
        'save_precision',
        'network_module',
        'network_dim',
        'learning_rate',
        'cache_text_encoder_outputs',
        'cache_text_encoder_outputs_to_disk',
        'fp8_base',
        'highvram',
        'max_train_epochs',
        'save_every_n_epochs',
        'dataset_config',
        'output_dir',
        'output_name',
        'timestep_sampling',
        'discrete_flow_shift',
        'model_prediction_type',
        'guidance_scale',
        'loss_type',
        'optimizer_type',
        'optimizer_args',
        'lr_scheduler',
        'sample_prompts',
        'sample_every_n_steps',
        'max_grad_norm',
        'split_mode',
        'network_args'
    }

    # generate a UI config
    # if not in basic_args, create a simple form
    parser = train_network.setup_parser()
    flux_train_utils.add_flux_train_arguments(parser)
    args_info = {}
    for action in parser._actions:
        if action.dest != 'help':  # Skip the default help argument
            # if the dest is included in basic_args
            args_info[action.dest] = {
                "action": action.option_strings,  # Option strings like '--use_8bit_adam'
                "type": action.type,              # Type of the argument
                "help": action.help,              # Help message
                "default": action.default,        # Default value, if any
                "required": action.required       # Whether the argument is required
            }
    temp = []
    for key in args_info:
        temp.append({ 'key': key, 'action': args_info[key] })
    temp.sort(key=lambda x: x['key'])
    advanced_component_ids = []
    advanced_components = []
    for item in temp:
        key = item['key']
        action = item['action']
        if key in basic_args:
            print("")
        else:
            action_type = str(action['type'])
            component = None
            with gr.Column(min_width=300):
                if action_type == "None":
                    # radio
                    component = gr.Checkbox()
    #            elif action_type == "<class 'str'>":
    #                component = gr.Textbox()
    #            elif action_type == "<class 'int'>":
    #                component = gr.Number(precision=0)
    #            elif action_type == "<class 'float'>":
    #                component = gr.Number()
    #            elif "int_or_float" in action_type:
    #                component = gr.Number()
                else:
                    component = gr.Textbox(value="")
                if component != None:
                    component.interactive = True
                    component.elem_id = action['action'][0]
                    component.label = component.elem_id
                    component.elem_classes = ["advanced"]
                if action['help'] != None:
                    component.info = action['help']
            advanced_components.append(component)
            advanced_component_ids.append(component.elem_id)
    return advanced_components, advanced_component_ids


theme="Kasien/ali_theme_custom"
css = """
@keyframes rotate {
    0% {
        transform: rotate(0deg);
    }
    100% {
        transform: rotate(360deg);
    }
}
#advanced_options .advanced:nth-child(even) { background: rgba(0,0,100,0.04) !important; }
h1{font-family: georgia; font-style: italic; font-weight: bold; font-size: 30px; letter-spacing: -1px;}
h3{margin-top: 0}
.tabitem{border: 0px}
.group_padding{}
nav{position: fixed; top: 0; left: 0; right: 0; z-index: 1000; text-align: center; padding: 10px; box-sizing: border-box; display: flex; align-items: center; backdrop-filter: blur(10px); }
nav button { background: none; color: firebrick; font-weight: bold; border: 2px solid firebrick; padding: 5px 10px; border-radius: 5px; font-size: 14px; }
nav img { height: 40px; width: 40px; border-radius: 40px; }
nav img.rotate { animation: rotate 2s linear infinite; }
.flexible { flex-grow: 1; }
.tast-details { margin: 10px 0 !important; }
.toast-wrap { bottom: var(--size-4) !important; top: auto !important; border: none !important; backdrop-filter: blur(10px); }
.toast-title, .toast-text, .toast-icon, .toast-close { color: black !important; font-size: 14px; }
.toast-body { border: none !important; }
#terminal { box-shadow: none !important; margin-bottom: 25px; background: rgba(0,0,0,0.03); }
#terminal .generating { border: none !important; }
#terminal label { position: absolute !important; }
.tabs { margin-top: 50px; }
.hidden { display: none !important; }
.codemirror-wrapper .cm-line { font-size: 12px !important; }
label { font-weight: bold !important; }
#start_training.clicked { background: silver; color: black; }
"""

js = """
function() {
    let autoscroll = document.querySelector("#autoscroll")
    if (window.iidxx) {
        window.clearInterval(window.iidxx);
    }
    window.iidxx = window.setInterval(function() {
        let text=document.querySelector(".codemirror-wrapper .cm-line").innerText.trim()
        let img = document.querySelector("#logo")
        if (text.length > 0) {
            autoscroll.classList.remove("hidden")
            if (autoscroll.classList.contains("on")) {
                autoscroll.textContent = "自动滚动 开"
                window.scrollTo(0, document.body.scrollHeight, { behavior: "smooth" });
                img.classList.add("rotate")
            } else {
                autoscroll.textContent = "自动滚动 关"
                img.classList.remove("rotate")
            }
        }
    }, 500);
    console.log("autoscroll", autoscroll)
    autoscroll.addEventListener("click", (e) => {
        autoscroll.classList.toggle("on")
    })
    function debounce(fn, delay) {
        let timeoutId;
        return function(...args) {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => fn(...args), delay);
        };
    }

    function handleClick() {
        console.log("刷新")
        document.querySelector("#refresh").click();
    }
    const debouncedClick = debounce(handleClick, 1000);
    document.addEventListener("input", debouncedClick);

    document.querySelector("#start_training").addEventListener("click", (e) => {
      e.target.classList.add("clicked")
      e.target.innerHTML = "训练中..."
    })

}
"""

current_account = account_hf()
print(f"current_account={current_account}")

with gr.Blocks(elem_id="app", theme=theme, css=css, fill_width=True) as demo:
    with gr.Tabs() as tabs:
        with gr.TabItem("健身房"):
            output_components = []
            with gr.Row():
                gr.HTML("""<nav>
            <img id='logo' src='/file=icon.png' width='80' height='80'>
            <div class='flexible'></div>
            <button id='autoscroll' class='on hidden'></button>
        </nav>
        """)
            with gr.Row(elem_id='container'):
                with gr.Column():
                    gr.Markdown(
                        """# 第一步. LoRA信息
        <p style="margin-top:0">配置你的LoRA训练设置。</p>
        """, elem_classes="group_padding")
                    lora_name = gr.Textbox(
                        label="你的LoRA名称",
                        info="这必须是一个唯一的名称",
                        placeholder="例如：波斯微型绘画风格，猫玩具",
                    )
                    concept_sentence = gr.Textbox(
                        elem_id="--concept_sentence",
                        label="触发词/句子",
                        info="要使用的触发词或句子",
                        placeholder="不常见的词如p3rs0n或trtcrd，或句子如'in the style of CNSTLL'",
                        interactive=True,
                    )
                    model_names = list(models.keys())
                    print(f"model_names={model_names}")
                    base_model = gr.Dropdown(label="基础模型（编辑models.yaml文件以添加更多到此列表）", choices=model_names, value=model_names[0])
                    vram = gr.Radio(["20G", "16G", "12G" ], value="20G", label="VRAM", interactive=True)
                    num_repeats = gr.Number(value=10, precision=0, label="每张图片重复训练次数", interactive=True)
                    max_train_epochs = gr.Number(label="最大训练周期", value=16, interactive=True)
                    total_steps = gr.Number(0, interactive=False, label="预计训练步数")
                    sample_prompts = gr.Textbox("", lines=5, label="样本图像提示（用新行分隔）", interactive=True)
                    sample_every_n_steps = gr.Number(0, precision=0, label="每N步生成样本图像", interactive=True)
                    resolution = gr.Number(value=512, precision=0, label="调整数据集图像大小")
                with gr.Column():
                    gr.Markdown(
                        """# 第二步. 数据集
        <p style="margin-top:0">确保标题包含触发词。</p>
        """, elem_classes="group_padding")
                    with gr.Group():
                        images = gr.File(
                            file_types=["image", ".txt"],
                            label="上传你的图片",
                            #info="如果你想，你也可以手动上传与图片名称匹配的标题文件（例如：img0.png => img0.txt）",
                            file_count="multiple",
                            interactive=True,
                            visible=True,
                            scale=1,
                        )
                    with gr.Group(visible=False) as captioning_area:
                        do_captioning = gr.Button("使用Florence-2添加AI标题")
                        output_components.append(captioning_area)
                        #output_components = [captioning_area]
                        caption_list = []
                        for i in range(1, MAX_IMAGES + 1):
                            locals()[f"captioning_row_{i}"] = gr.Row(visible=False)
                            with locals()[f"captioning_row_{i}"]:
                                locals()[f"image_{i}"] = gr.Image(
                                    type="filepath",
                                    width=111,
                                    height=111,
                                    min_width=111,
                                    interactive=False,
                                    scale=2,
                                    show_label=False,
                                    show_share_button=False,
                                    show_download_button=False,
                                )
                                locals()[f"caption_{i}"] = gr.Textbox(
                                    label=f"标题 {i}", scale=15, interactive=True
                                )

                            output_components.append(locals()[f"captioning_row_{i}"])
                            output_components.append(locals()[f"image_{i}"])
                            output_components.append(locals()[f"caption_{i}"])
                            caption_list.append(locals()[f"caption_{i}"])
                with gr.Column():
                    gr.Markdown(
                        """# 第三步. 训练
        <p style="margin-top:0">按开始开始训练。</p>
        """, elem_classes="group_padding")
                    refresh = gr.Button("刷新", elem_id="refresh", visible=False)
                    start = gr.Button("开始训练", visible=False, elem_id="start_training")
                    output_components.append(start)
                    train_script = gr.Textbox(label="训练脚本", max_lines=100, interactive=True)
                    train_config = gr.Textbox(label="训练配置", max_lines=100, interactive=True)
            with gr.Accordion("高级选项", elem_id='advanced_options', open=False):
                with gr.Row():
                    with gr.Column(min_width=300):
                        seed = gr.Number(label="--seed", info="种子", value=42, interactive=True)
                    with gr.Column(min_width=300):
                        workers = gr.Number(label="--max_data_loader_n_workers", info="工作线程数量", value=2, interactive=True)
                    with gr.Column(min_width=300):
                        learning_rate = gr.Textbox(label="--learning_rate", info="学习率", value="8e-4", interactive=True)
                    with gr.Column(min_width=300):
                        save_every_n_epochs = gr.Number(label="--save_every_n_epochs", info="每N个周期保存一次", value=4, interactive=True)
                    with gr.Column(min_width=300):
                        guidance_scale = gr.Number(label="--guidance_scale", info="引导比例", value=1.0, interactive=True)
                    with gr.Column(min_width=300):
                        timestep_sampling = gr.Textbox(label="--timestep_sampling", info="时间步采样", value="shift", interactive=True)
                    with gr.Column(min_width=300):
                        network_dim = gr.Number(label="--network_dim", info="LoRA等级", value=4, minimum=4, maximum=128, step=4, interactive=True)
                    advanced_components, advanced_component_ids = init_advanced()
            with gr.Row():
                terminal = LogsView(label="训练日志", elem_id="terminal")
            with gr.Row():
                gallery = gr.Gallery(get_samples, inputs=[lora_name], label="样本", every=10, columns=6)

        with gr.TabItem("发布") as publish_tab:
            hf_token = gr.Textbox(label="Huggingface Token")
            hf_login = gr.Button("登录")
            hf_logout = gr.Button("登出")
            with gr.Row() as row:
                gr.Markdown("**LoRA**")
                gr.Markdown("**上传**")
            loras = get_loras()
            with gr.Row():
                lora_rows = refresh_publish_tab()
                with gr.Column():
                    with gr.Row():
                        repo_owner = gr.Textbox(label="账户", interactive=False)
                        repo_name = gr.Textbox(label="仓库名称")
                    repo_visibility = gr.Textbox(label="仓库可见性（'public' 或 'private'）", value="public")
                    upload_button = gr.Button("上传到HuggingFace")
                    upload_button.click(
                        fn=upload_hf,
                        inputs=[
                            base_model,
                            lora_rows,
                            repo_owner,
                            repo_name,
                            repo_visibility,
                            hf_token,
                        ]
                    )
            hf_login.click(fn=login_hf, inputs=[hf_token], outputs=[hf_token, hf_login, hf_logout, repo_owner])
            hf_logout.click(fn=logout_hf, outputs=[hf_token, hf_login, hf_logout, repo_owner])


    publish_tab.select(refresh_publish_tab, outputs=lora_rows)
    lora_rows.select(fn=set_repo, inputs=[lora_rows], outputs=[repo_name])

    dataset_folder = gr.State()

    listeners = [
        base_model,
        lora_name,
        resolution,
        seed,
        workers,
        concept_sentence,
        learning_rate,
        network_dim,
        max_train_epochs,
        save_every_n_epochs,
        timestep_sampling,
        guidance_scale,
        vram,
        num_repeats,
        sample_prompts,
        sample_every_n_steps,
        *advanced_components
    ]
    advanced_component_ids = [x.elem_id for x in advanced_components]
    original_advanced_component_values = [comp.value for comp in advanced_components]
    images.upload(
        load_captioning,
        inputs=[images, concept_sentence],
        outputs=output_components
    )
    images.delete(
        load_captioning,
        inputs=[images, concept_sentence],
        outputs=output_components
    )
    images.clear(
        hide_captioning,
        outputs=[captioning_area, start]
    )
    max_train_epochs.change(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps]
    )
    num_repeats.change(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps]
    )
    images.upload(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps]
    )
    images.delete(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps]
    )
    images.clear(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps]
    )
    concept_sentence.change(fn=update_sample, inputs=[concept_sentence], outputs=sample_prompts)
    start.click(fn=create_dataset, inputs=[dataset_folder, resolution, images] + caption_list, outputs=dataset_folder).then(
        fn=start_training,
        inputs=[
            base_model,
            lora_name,
            train_script,
            train_config,
            sample_prompts,
        ],
        outputs=terminal,
    )
    do_captioning.click(fn=run_captioning, inputs=[images, concept_sentence] + caption_list, outputs=caption_list)
    demo.load(fn=loaded, js=js, outputs=[hf_token, hf_login, hf_logout, repo_owner])
    refresh.click(update, inputs=listeners, outputs=[train_script, train_config, dataset_folder])
if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    demo.launch(server_name='0.0.0.0', debug=True, show_error=True, allowed_paths=[cwd])

import gradio as gr
import torch
from TTS.api import TTS
from TTS.utils.radam import RAdam
from torch.serialization import add_safe_globals
import whisper
import subprocess
import os
import requests
from opencc import OpenCC

from utils.CutVoice import trim_tail_by_energy_and_gradient

# 正确注册 RAdam 类
add_safe_globals({"RAdam": RAdam})

cc = OpenCC('t2s')

# 注册其他可能需要的 TTS 类（避免 torch.load 报错）
def safe_register_all_globals():
    torch.serialization._allowed_globals = {
        "__builtin__": set(dir(__builtins__)),
        "builtins": set(dir(__builtins__)),
    }
    torch.serialization._allowed_globals.update({
        "TTS.utils.radam": {"RAdam"},
        "TTS.vocoder.utils.generic": {"ADAM"},
        "TTS.models.tacotron.loss": {"TacotronLoss"},
        "TTS.vocoder.models.wavernn": {"Wavernn"},
    })

safe_register_all_globals()

# 初始化 TTS 模型
MODEL_NAME = "tts_models/zh-CN/baker/tacotron2-DDC-GST"
tts = TTS(model_name=MODEL_NAME, progress_bar=True, gpu=False)

# 初始化 Whisper 模型 (ASR)
asr_model = whisper.load_model("base")

# 模型自动下载器（for SadTalker）
def download_models():
    model_list = [
        ("shape_predictor_68_face_landmarks.dat", "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/shape_predictor_68_face_landmarks.dat", "checkpoints"),
        ("wav2lip.pth", "https://huggingface.co/guoyww/facevid2vid/resolve/main/wav2lip.pth", "checkpoints"),
        ("mapping_00109-model.pth.tar", "https://huggingface.co/guoyww/facevid2vid/resolve/main/mapping_00109-model.pth.tar", "checkpoints"),
        ("parsing_model.pth", "https://huggingface.co/guoyww/facevid2vid/resolve/main/parsing_model.pth", "checkpoints"),
        ("GFPGANv1.4.pth", "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.8/GFPGANv1.4.pth", "checkpoints/gfpgan")
    ]

    for name, url, folder in model_list:
        os.makedirs(folder, exist_ok=True)
        dest = os.path.join(folder, name)
        if os.path.exists(dest):
            print(f"[已存在] {name}")
            continue
        print(f"[下载中] {name} ...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"[完成] {dest}")

# 合成语音
def generate_speech(text):
    output_path = "output.wav"
    tts.tts_to_file(text=text, file_path=output_path)
    trim_tail_by_energy_and_gradient(output_path)
    return output_path

# 语音转文字
def transcribe_audio(audio_file):
    if not audio_file:
        return "⚠️ 没有上传任何音频文件"

    try:
        # 确保上传的音频文件已写入且不为空
        if not os.path.exists(audio_file):
            return "❗ 找不到音频文件，请重新上传"
        if os.path.getsize(audio_file) < 2048:
            return "⚠️ 音频文件太小，可能无效或上传不完整"

        # 使用 Whisper 识别并转换为简体中文
        result = asr_model.transcribe(audio_file, language="zh")
        simplified = cc.convert(result["text"])
        return simplified
    except Exception as e:
        return f"识别失败：{str(e)}"

# 生成数字人动画（使用 SadTalker 的 launcher.py）
def generate_video(image_path, audio_path):
    if not image_path or not os.path.exists(image_path):
        return "⚠️ 没有上传头像图片或文件不存在"
    if os.path.getsize(image_path) < 2048:
        return "⚠️ 上传的头像文件太小，可能无效"

    if not audio_path or not os.path.exists(audio_path):
        return "⚠️ 没有上传音频文件或文件不存在"
    if os.path.getsize(audio_path) < 2048:
        return "⚠️ 音频文件太小，可能无效或上传不完整"

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    launcher_path = os.path.abspath("sadtalker/launcher.py")
    cmd = [
        "python", launcher_path,
        "--driven_audio", audio_path,
        "--source_image", image_path,
        "--result_dir", output_dir,
        "--preprocess", "full",
        "--still",
        "--enhancer", "gfpgan"
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        return f"生成失败：\n命令：{' '.join(e.cmd)}\n返回码：{e.returncode}"

    output_video_path = os.path.join(output_dir, "result.mp4")
    return output_video_path if os.path.exists(output_video_path) else "生成失败，未找到视频文件"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🎤 本地数字人全功能工具")

    with gr.Tab("文字转语音"):
        text_input = gr.Textbox(label="输入文字")
        generate_btn = gr.Button("🎧 合成语音")
        output_audio = gr.Audio(label="语音文件", type="filepath")
        generate_btn.click(fn=generate_speech, inputs=text_input, outputs=output_audio)

    with gr.Tab("语音转文字"):
        audio_input = gr.Audio(label="上传语音", type="filepath")
        transcribe_btn = gr.Button("📑 识别")
        asr_output = gr.Textbox(label="识别结果")
        transcribe_btn.click(fn=transcribe_audio, inputs=audio_input, outputs=asr_output)

    with gr.Tab("数字人动画"):
        image_input = gr.Image(label="上传头像", type="filepath")
        driven_audio_input = gr.Audio(label="使用合成或自己语音", type="filepath")
        generate_video_btn = gr.Button("🎥 生成动画")
        video_output = gr.Video(label="数字人视频")
        generate_video_btn.click(fn=generate_video, inputs=[image_input, driven_audio_input], outputs=video_output)

    with gr.Tab("模型下载"):
        gr.Markdown("### 🧩 首次使用请点击下载 SadTalker 模型")
        download_btn = gr.Button("📥 下载模型")
        download_output = gr.Textbox(label="状态输出")
        download_btn.click(fn=download_models, outputs=download_output)

demo.launch()

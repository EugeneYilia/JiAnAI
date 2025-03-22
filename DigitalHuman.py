import gradio as gr
import torch
from TTS.api import TTS
from TTS.utils.radam import RAdam
from torch.serialization import add_safe_globals
import whisper
from utils.CutVoice import trim_tail_by_energy_and_gradient
import subprocess
import os

# 正确注册 RAdam 类
add_safe_globals({"RAdam": RAdam})

# 兼容 TTS 模型反序列化中引用的其他类
import types

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
asr_model = whisper.load_model("base")  # 可换成 tiny/small/medium/large

# 合成语音
def generate_speech(text):
    output_path = "output.wav"
    tts.tts_to_file(text=text, file_path=output_path)
    trim_tail_by_energy_and_gradient(output_path)
    return output_path

# 语音转文字
def transcribe_audio(audio_file):
    if audio_file is None:
        return "请先上传语音文件"
    result = asr_model.transcribe(audio_file, language="zh")
    return result["text"]

# 生成数字人动画（使用 SadTalker 的 launcher.py）
def generate_video(image_path, audio_path):
    if image_path is None or audio_path is None:
        return "请上传图片和配音"
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, "result.mp4")
    cmd = [
        "python", "sadtalker/launcher.py",
        "--driven_audio", audio_path,
        "--source_image", image_path,
        "--result_dir", output_dir,
        "--preprocess", "full",
        "--still",
        "--enhancer", "gfpgan"
    ]
    subprocess.run(cmd, check=True)
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

demo.launch()
import gradio as gr
import torch
from TTS.api import TTS
from TTS.utils.radam import RAdam
from torch.serialization import add_safe_globals
import whisper

from utils.CutVoice import trim_tail_by_energy_and_gradient

# ✅ 正确注册自定义类（适配 PyTorch 2.6）
add_safe_globals({RAdam.__name__: RAdam})

# 模型名称
MODEL_NAME = "tts_models/zh-CN/baker/tacotron2-DDC-GST"

# 初始化模型（使用 CPU）
tts = TTS(model_name=MODEL_NAME, progress_bar=True, gpu=False)

# 初始化 Whisper（语音转文字）
asr_model = whisper.load_model("large")  # 可改成 "small" / "medium" / "large"

# 合成函数
def generate_speech(text):
    output_path = "output.wav"
    tts.tts_to_file(text=text, file_path=output_path)
    trim_tail_by_energy_and_gradient(output_path)
    return output_path

# 语音识别函数
# 这里使用 Whisper 模型进行语音转文字识别
# 注意：Whisper 模型需要一定的计算资源，可能会比较慢
# 可以根据需要选择不同大小的模型（"tiny" / "base" / "small" / "medium" / "large"）
def transcribe_audio(audio_file):
    if audio_file is None:
        return "请先上传语音文件"
    result = asr_model.transcribe(audio_file, language="zh")
    return result["text"]

# 创建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("## 🎙️ 本地文字转语音工具（中文）")
    with gr.Row():
        text_input = gr.Textbox(label="请输入文字", lines=3, placeholder="比如：你好，欢迎使用语音合成工具。")
    with gr.Row():
        generate_btn = gr.Button("🎧 生成语音")
        output_audio = gr.Audio(label="合成结果", type="filepath")

    generate_btn.click(fn=generate_speech, inputs=text_input, outputs=output_audio)

    # 分割
    gr.Markdown("---")

    # 语音转文本
    with gr.Group():
        gr.Markdown("### 📝 语音转文字")
        audio_input = gr.Audio(label="上传语音文件（支持中文）", type="filepath")
        transcribe_btn = gr.Button("📝 识别语音文字")
        asr_output = gr.Textbox(label="识别结果")

        transcribe_btn.click(fn=transcribe_audio, inputs=audio_input, outputs=asr_output)

# 启动 Gradio 服务
demo.launch()

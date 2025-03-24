import os
import subprocess
import warnings
import zipfile
from datetime import datetime

import gradio as gr
import jieba
import requests
import torch
import whisper
from TTS.api import TTS
from TTS.utils.radam import RAdam
from fastapi.staticfiles import StaticFiles
from opencc import OpenCC
from torch.serialization import add_safe_globals

from utils.CutVoice import trim_tail_by_energy_and_gradient

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
jieba.setLogLevel(jieba.logging.WARN)

RECOGNIZED_DIR = "recognized"
os.makedirs(RECOGNIZED_DIR, exist_ok=True)

RECOGNIZED_EXPORT_DIR = "recognized_export"
os.makedirs(RECOGNIZED_EXPORT_DIR, exist_ok=True)

# 注册 RAdam 类
add_safe_globals({"RAdam": RAdam})
cc = OpenCC('t2s')

############################################################################
# 将背景图片 URL 写成单行，并使用 no-repeat center center
############################################################################
material_css = """
@import url('https://fonts.googleapis.com/css?family=Roboto:400,500,700&display=swap');

:root {
  --md-primary: #1976d2;
  --md-primary-dark: #1565c0;
  --md-secondary: #424242;
  --md-background: #f9f9f9;
  --md-surface: #ffffff;
  --md-text: #212121;
  --md-text-on-primary: #ffffff;
  --md-border-radius: 8px;
  --md-transition: 0.3s ease;
}

/* 整个页面的背景：底层是固定(fixed)的图片 + 半透明渐变 */
html, body, .gradio-container {
  margin: 0;
  padding: 0;
  font-family: 'Roboto', sans-serif;
  color: var(--md-text);

  /* 多层背景：先图片，后渐变。都用 cover 覆盖屏幕，不随内容滚动 */
  background:
    url("https://raw.githubusercontent.com/EugeneYilia/JiAnAI/master/assets/images/freemasonry.png")
      no-repeat center center fixed,
    linear-gradient(rgba(255,255,255,0.4), rgba(255,255,255,0.4)) /* 半透明 */
      no-repeat fixed;

  background-size: cover, cover;
  background-color: transparent !important;
}

/* 让 Tabs、Box 等主要容器有更高的不透明度背景，以便文字清晰 */
.tabs, .tabitem, .gr-box, .gr-group, .gr-row, .gr-column {
  background-color: rgba(255, 255, 255, 0.9) !important; /* 0.9 不透明度 */
  border-radius: var(--md-border-radius) !important;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  margin-top: 8px !important; /* 让容器与背景分离一点 */
  padding: 12px !important;   /* 内间距让文字更舒展 */
}

/* 输入区域、文件上传、音频组件等，底色稍微提高亮度 */
.gr-textbox, .gr-file, .gr-audio {
  background-color: #ffffff !important; 
  border-radius: var(--md-border-radius) !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  padding: 8px !important;
}

.gr-textbox textarea {
  min-height: 100px !important;
  background-color: #fff !important;
  border: 1px solid #ccc !important;
  border-radius: var(--md-border-radius) !important;
}

/* 调整按钮风格，维持 Material 风格 */
button, .gr-button {
  background-color: var(--md-primary) !important;
  color: var(--md-text-on-primary) !important;
  border: none !important;
  border-radius: var(--md-border-radius) !important;
  font-weight: 500;
  cursor: pointer;
  transition: background-color var(--md-transition), box-shadow var(--md-transition);
  box-shadow: 0 2px 4px rgba(0,0,0,0.15);
  padding: 0.6em 1.2em !important;
}
button:hover, .gr-button:hover {
  background-color: var(--md-primary-dark) !important;
  box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

/* 让 Tab 标题有更明显的选中状态 */
.tabs button {
  background: transparent !important;
  color: var(--md-secondary) !important;
  border: none !important;
  box-shadow: none !important;
  border-radius: 0 !important;
  font-weight: 500 !important;
  padding: 0.6em 1.2em !important;
  transition: background-color var(--md-transition), color var(--md-transition);
}
.tabs button.selected {
  color: var(--md-primary) !important;
  border-bottom: 3px solid var(--md-primary) !important;
  background-color: transparent !important;
}

/* Footer / share 按钮区域居中 */
.footer, .share-link-container {
  text-align: center !important;
  margin-top: 20px;
}


"""

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

MODEL_NAME = "tts_models/zh-CN/baker/tacotron2-DDC-GST"
tts = TTS(model_name=MODEL_NAME, progress_bar=True, gpu=False)
asr_model = whisper.load_model("large")

def download_models():
    model_list = [
        ("shape_predictor_68_face_landmarks.dat",
         "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/shape_predictor_68_face_landmarks.dat",
         "checkpoints"),
        ("wav2lip.pth", "https://huggingface.co/guoyww/facevid2vid/resolve/main/wav2lip.pth", "checkpoints"),
        ("mapping_00109-model.pth.tar",
         "https://huggingface.co/guoyww/facevid2vid/resolve/main/mapping_00109-model.pth.tar", "checkpoints"),
        ("parsing_model.pth", "https://huggingface.co/guoyww/facevid2vid/resolve/main/parsing_model.pth", "checkpoints"),
        ("GFPGANv1.4.pth", "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.8/GFPGANv1.4.pth",
         "checkpoints/gfpgan")
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

def generate_speech(text):
    output_path = "output.wav"
    tts.tts_to_file(text=text, file_path=output_path)
    trim_tail_by_energy_and_gradient(output_path)
    return output_path

def transcribe_audio(audio_file):
    if not audio_file:
        return "⚠️ 没有上传任何音频文件"
    try:
        if not os.path.exists(audio_file):
            return "❗ 找不到音频文件，请重新上传"
        if os.path.getsize(audio_file) < 2048:
            return "⚠️ 音频文件太小，可能上传不完整或为空，请重新上传"

        import soundfile as sf
        try:
            _ = sf.info(audio_file)
        except Exception:
            return "⚠️ 音频文件格式不支持或内容损坏，请重新上传"

        result = asr_model.transcribe(audio_file, language="zh")
        simplified = ""
        for char in result["text"]:
            if char in "。！？；，、,.!?;:":
                simplified += char
            else:
                simplified += cc.convert(char)
        save_recognition_history(result["text"], simplified)
        return simplified
    except Exception as e:
        raise e

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

def save_recognition_history(text_raw, text_simplified):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_txt = os.path.join(RECOGNIZED_DIR, f"recognized_{timestamp}.txt")
    filename_docx = os.path.join(RECOGNIZED_DIR, f"recognized_{timestamp}.docx")

    with open(filename_txt, "w", encoding="utf-8") as f:
        f.write(f"[识别时间] {timestamp}\n")
        f.write(f"[原始文本]\n{text_raw}\n\n")
        f.write(f"[简体结果]\n{text_simplified}\n")

    from docx import Document
    doc = Document()
    doc.add_heading("语音识别结果", level=1)
    doc.add_paragraph(f"识别时间: {timestamp}")
    doc.add_paragraph("原始文本:")
    doc.add_paragraph(text_raw)
    doc.add_paragraph("转换为简体：")
    doc.add_paragraph(text_simplified)
    doc.save(filename_docx)

def export_recognition_zip():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = f"recognized_export/recognized_export_{timestamp}.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for filename in os.listdir(RECOGNIZED_DIR):
            file_path = os.path.join(RECOGNIZED_DIR, filename)
            zipf.write(file_path, arcname=filename)
    return zip_path

def search_history_by_question(query):
    hits = []
    for filename in os.listdir(RECOGNIZED_DIR):
        path = os.path.join(RECOGNIZED_DIR, filename)
        if filename.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                if query in content:
                    hits.append(f"📄 {filename}\n{content[:300]}...\n---")
    if not hits:
        return "未找到相关内容。请尝试输入更常见的关键词。"
    return "\n\n".join(hits)

demo_config = gr.Blocks(css=material_css)

with demo_config as demo:
    gr.Markdown("## 🎤 吉安智能体")

    with gr.Tab("文字转语音"):
        text_input = gr.Textbox(label="输入文字")
        generate_btn = gr.Button("🎧 合成语音")
        output_audio = gr.Audio(label="语音文件", type="filepath", interactive=True)
        generate_btn.click(fn=generate_speech, inputs=text_input, outputs=output_audio)

    with gr.Tab("语音转文字"):
        with gr.Row():
            audio_input = gr.File(label="上传语音 (仅限 WAV 格式)", file_types=[".wav"], interactive=True)
            upload_status = gr.Textbox(label="语音上传状态", interactive=False, max_lines=1,
                                       container=True, show_copy_button=True)
        transcribe_btn = gr.Button("📑 识别")
        asr_output = gr.Textbox(label="识别结果")

        def check_audio_upload_status(audio_file):
            if isinstance(audio_file, str) and os.path.exists(audio_file) and os.path.getsize(audio_file) > 2048 and audio_file.endswith('.wav'):
                return "✅ 音频上传完成"
            return "⚠️ 音频文件过小或上传失败"

        audio_input.change(fn=check_audio_upload_status, inputs=audio_input, outputs=upload_status)
        transcribe_btn.click(fn=transcribe_audio, inputs=audio_input, outputs=asr_output)

    with gr.Tab("数字人动画"):
        with gr.Row():
            with gr.Column():
                image_input = gr.File(label="上传头像 (PNG/JPG)", file_types=[".png", ".jpg", ".jpeg"], interactive=True)
                image_name = gr.Textbox(label="头像文件名", interactive=False, max_lines=1)
                image_status = gr.Textbox(label="头像上传状态", interactive=False, max_lines=1,
                                          container=True, show_copy_button=True)
                image_preview = gr.Image(label="头像预览", interactive=False)

                def update_image_preview(image_file):
                    if not image_file or not os.path.exists(image_file):
                        return gr.update(visible=False)
                    if os.path.getsize(image_file) < 2048 or not image_file.lower().endswith((".png", ".jpg", ".jpeg")):
                        return gr.update(visible=False)
                    return gr.update(value=image_file, visible=True)

                image_input.change(fn=update_image_preview, inputs=image_input, outputs=image_preview)

        with gr.Row():
            with gr.Column():
                driven_audio_input = gr.Audio(label="使用合成或自己语音", type="filepath",
                                              interactive=True, show_label=True,
                                              sources=["upload"], format="wav")
                audio_name = gr.Textbox(label="音频文件名", interactive=False, max_lines=1)
                audio_status = gr.Textbox(label="音频上传状态", interactive=False, max_lines=1,
                                          container=True, show_copy_button=True)

        generate_video_btn = gr.Button("🎥 生成动画")
        video_output = gr.Video(label="数字人视频")

        def check_image_upload_status(image_file):
            if isinstance(image_file, str) and os.path.exists(image_file) and os.path.getsize(image_file) > 2048 and image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                return "✅ 头像上传完成"
            return "⚠️ 头像文件过小或上传失败"

        def check_audio_upload_status_generic(audio_file):
            if audio_file and os.path.exists(audio_file) and os.path.getsize(audio_file) > 2048:
                return "✅ 音频上传完成"
            return "⚠️ 音频文件过小或上传失败"

        image_input.change(fn=lambda f: os.path.basename(f) if f else "未选择文件", inputs=image_input, outputs=image_name)
        image_input.change(fn=check_image_upload_status, inputs=image_input, outputs=image_status)
        driven_audio_input.change(fn=lambda f: os.path.basename(f) if f else "未选择文件", inputs=driven_audio_input, outputs=audio_name)
        driven_audio_input.change(fn=check_audio_upload_status_generic, inputs=driven_audio_input, outputs=audio_status)

        generate_video_btn.click(fn=generate_video, inputs=[image_input, driven_audio_input], outputs=video_output)

    with gr.Tab("模型下载"):
        gr.Markdown("### 🧩 首次使用请点击下载 SadTalker 模型")
        download_btn = gr.Button("📥 下载模型")
        download_output = gr.Textbox(label="状态输出")
        download_btn.click(fn=download_models, outputs=download_output)

    with gr.Tab("识别历史"):
        gr.Markdown("### 📄 导出历史 / 查询内容")

        with gr.Row():
            export_btn = gr.Button("📦 导出 ZIP")
            export_file = gr.File(label="下载识别记录压缩包")
            export_btn.click(fn=export_recognition_zip, outputs=export_file)

        with gr.Row():
            query_input = gr.Textbox(label="输入关键词或内容问题")
            query_btn = gr.Button("🔍 查询记录")
            query_result = gr.Textbox(label="查询结果", lines=8)
            query_btn.click(fn=search_history_by_question, inputs=query_input, outputs=query_result)

demo.launch(share=False)

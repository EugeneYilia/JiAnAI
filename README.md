# 🎤 本地部署·中文数字人系统（AI 虚拟人语音工具）

语音合成 · 语音识别 · 虚拟人动画生成 · Whisper · SadTalker · TTS · 全流程本地部署 · 高质量中文支持 ✅

---

## ✨ 功能一览

| 模块                   | 支持情况 |
|------------------------|----------|
| 🗣️ 文本转语音（TTS）        | ✅ Tacotron2 + 后处理剪尾音 |
| 🎧 语音转文字（ASR）        | ✅ Whisper Large + 繁转简 + 保留标点 |
| 🧑‍🎤 数字人视频生成（SadTalker） | ✅ 上传头像 + 音频生成唇形同步视频 |
| 🖼️ 图像预览 & 音频自动播放    | ✅ 上传后立即可预览 / 播放 |
| 📄 自动保存识别历史记录       | ✅ 支持 txt / docx / zip 导出 |
| 💬 对话式语义检索识别历史     | ✅ 支持自然语言提问“我说过哪些智慧城市？” |
| ☁️ 飞书 / Notion 同步识别内容  | ✅ 支持配置 Webhook 自动推送结果 |

---

## 🚀 快速启动

```bash
git clone https://github.com/EugeneYilia/JiAnAI
cd JiAnAI

# 安装依赖（推荐 Python ≥ 3.9）
pip install -r requirements.txt

# 启动系统
python DigitalHuman.py
```

浏览器打开： [http://127.0.0.1:7860](http://127.0.0.1:7860)

---

## 🖥️ 界面展示

- ✅ 上传头像图片预览  
- ✅ 上传语音自动播放  
- ✅ 点击生成即可合成唇形动画  
- ✅ 语音识别支持复制、语义搜索、历史管理

---

## 📦 项目结构

```
├── main.py                   # 主程序入口
├── utils/
│   └── CutVoice.py          # 音频剪尾处理
├── static/audio/            # 自动播放音频缓存目录
├── checkpoints/             # SadTalker 模型文件
├── results/                 # 生成的视频文件
├── recognized/              # 识别结果（txt / docx）
├── uploads/
│   ├── avatars/             # 上传头像
│   └── audio/               # 上传音频
├── requirements.txt         # 依赖
```

---

## 🧠 示例语义问答

- “我说过哪些跟智慧交通有关的内容？”
- “请总结我讲过的关于人工智能的观点”
- “查一下我提到‘数据安全’的记录有哪些”

> ✅ 系统会自动搜索匹配历史识别文本，并进行总结性回应。

---

## 🔌 可选平台同步配置

支持通过 Webhook 将识别/问答结果自动同步至：

- Notion 页面
- 飞书机器人
- 企业微信群机器人

---

## 📋 推荐依赖版本（requirements.txt）

```txt
absl-py==2.2.0
aiofiles==23.2.1
aiohappyeyeballs==2.6.1
aiohttp==3.11.14
aiosignal==1.3.2
annotated-types==0.7.0
anyascii==0.3.2
anyio==4.9.0
async-timeout==5.0.1
attrs==25.3.0
audioread==3.0.1
babel==2.17.0
bangla==0.0.2
blinker==1.9.0
blis==1.2.0
bnnumerizer==0.0.2
bnunicodenormalizer==0.1.7
catalogue==2.0.10
certifi==2025.1.31
cffi==1.17.1
charset-normalizer==3.4.1
click==8.1.8
cloudpathlib==0.21.0
colorama==0.4.6
confection==0.1.5
contourpy==1.2.1
coqpit==0.0.17
cycler==0.12.1
cymem==2.0.11
Cython==3.0.12
dateparser==1.1.8
decorator==5.2.1
docopt==0.6.2
einops==0.8.1
encodec==0.1.1
exceptiongroup==1.2.2
face-alignment==1.4.1
fastapi==0.115.11
ffmpeg-python==0.2.0
ffmpy==0.5.0
filelock==3.18.0
Flask==3.1.0
flatbuffers==25.2.10
fonttools==4.56.0
frozenlist==1.5.0
fsspec==2025.3.0
future==1.0.0
g2pkk==0.1.2
gradio==5.22.0
gradio_client==1.8.0
groovy==0.1.2
grpcio==1.71.0
gruut==2.2.3
gruut-ipa==0.13.0
gruut_lang_de==2.0.1
gruut_lang_en==2.0.1
gruut_lang_es==2.0.1
gruut_lang_fr==2.0.2
h11==0.14.0
hangul-romanize==0.1.0
httpcore==1.0.7
httpx==0.28.1
huggingface-hub==0.29.3
idna==3.10
imageio==2.37.0
imageio-ffmpeg==0.6.0
inflect==7.5.0
itsdangerous==2.2.0
jamo==0.4.1
jax==0.4.30
jaxlib==0.4.30
jieba==0.42.1
Jinja2==3.1.6
joblib==1.4.2
jsonlines==1.2.0
kiwisolver==1.4.8
langcodes==3.5.0
language_data==1.3.0
lazy_loader==0.4
librosa==0.10.0
llvmlite==0.43.0
marisa-trie==1.2.1
Markdown==3.7
markdown-it-py==3.0.0
MarkupSafe==3.0.2
matplotlib==3.8.4
mdurl==0.1.2
mediapipe==0.10.21
ml_dtypes==0.5.1
more-itertools==10.6.0
mpmath==1.3.0
msgpack==1.1.0
multidict==6.2.0
murmurhash==1.0.12
mutagen==1.47.0
networkx==2.8.8
nltk==3.9.1
num2words==0.5.14
numba==0.60.0
numpy==1.22.0
openai-whisper==20240930
opencc-python-reimplemented==0.1.7
opencv-contrib-python==4.11.0.86
opencv-python==4.11.0.86
opt_einsum==3.4.0
orjson==3.10.15
packaging==24.2
pandas==1.5.3
pillow==11.1.0
platformdirs==4.3.7
pooch==1.8.2
preshed==3.0.9
propcache==0.3.0
protobuf==4.25.6
psutil==7.0.0
pycparser==2.22
pydantic==2.10.6
pydantic_core==2.27.2
pydub==0.25.1
Pygments==2.19.1
pynndescent==0.5.13
pyparsing==3.2.1
pypinyin==0.53.0
pysbd==0.3.4
python-crfsuite==0.9.11
python-dateutil==2.9.0.post0
python-multipart==0.0.20
pytz==2025.1
PyYAML==6.0.2
regex==2024.11.6
requests==2.32.3
rich==13.9.4
ruff==0.11.2
safehttpx==0.1.6
safetensors==0.5.3
scikit-image==0.22.0
scikit-learn==1.6.1
scipy==1.11.4
semantic-version==2.10.0
sentencepiece==0.2.0
shellingham==1.5.4
six==1.17.0
smart-open==7.1.0
sniffio==1.3.1
sounddevice==0.5.1
soundfile==0.13.1
soxr==0.5.0.post1
spacy==3.8.4
spacy-legacy==3.0.12
spacy-loggers==1.0.5
srsly==2.5.1
starlette==0.46.1
SudachiDict-core==20250129
SudachiPy==0.6.10
sympy==1.13.1
tensorboard==2.19.0
tensorboard-data-server==0.7.2
thinc==8.3.4
threadpoolctl==3.6.0
tifffile==2025.3.13
tiktoken==0.9.0
tokenizers==0.21.1
tomlkit==0.13.2
torch==2.5.1
torchaudio==2.5.1
torchvision==0.13.1+cu113
tqdm==4.67.1
trainer==0.0.36
transformers==4.50.0
TTS @ git+https://github.com/coqui-ai/TTS@dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e
typeguard==4.4.2
typer==0.15.2
typing_extensions==4.12.2
tzdata==2025.1
tzlocal==5.3.1
umap-learn==0.5.7
Unidecode==1.3.8
urllib3==2.3.0
uvicorn==0.34.0
wasabi==1.1.3
weasel==0.4.1
websockets==15.0.1
Werkzeug==3.1.3
wrapt==1.17.2
yarl==1.18.3

```

---

## 🙏 感谢支持的开源项目

- [Whisper (OpenAI)](https://github.com/openai/whisper)
- [SadTalker](https://github.com/OpenTalker/SadTalker)
- [Coqui TTS](https://github.com/coqui-ai/TTS)
- [Gradio](https://github.com/gradio-app/gradio)

---
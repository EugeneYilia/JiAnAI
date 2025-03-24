import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()

# 获取绝对路径，确保与脚本同级或你指定的目录下有 assets/images/freemasonry.jpg
images_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "images")

# 打印出挂载路径和其中的文件列表，便于调试
print("Mounting path =", images_path)
try:
    print("Files in images folder =", os.listdir(images_path))
except FileNotFoundError:
    print("ERROR: assets/images folder not found!")
    # 你可以在这里抛出异常或继续

# 将 /assets/images 映射到本地目录 images_path
app.mount("/assets/images", StaticFiles(directory=images_path), name="images")

@app.get("/")
def read_root():
    return {"message": "Hello, world!"}

if __name__ == "__main__":
    # 启动测试服务
    uvicorn.run(app, host="127.0.0.1", port=8000)
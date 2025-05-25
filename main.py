# ✅ main.py
from flask import Flask, request, render_template
from PIL import Image
import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import requests
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

# 環境變數
load_dotenv()
app = Flask(__name__)

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# CLIP 設定
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Spotify 權杖
def get_spotify_token():
    r = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
    )
    return r.json().get("access_token")

# 圖片處理
def get_image_features(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711))
    ])
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        return model.get_image_features(image)

# 圖像情緒

def get_top_emotions(image):
    emotion_labels = [
        "joyful", "cheerful", "melancholic", "gloomy", "furious", "relaxed",
        "peaceful", "romantic", "dreamy", "mysterious", "eerie", "vibrant",
        "intense", "anxious", "nostalgic", "sentimental", "chill", "lo-fi",
        "energetic", "calm", "hopeful", "lonely"
    ]
    prompts = [f"This image evokes a {mood} feeling" for mood in emotion_labels]
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    image_features = get_image_features(image)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
    top_indices = similarity.topk(3).indices
    return [emotion_labels[i] for i in top_indices]

# 使用 Ollama

def generate_emotion_description(emotions):
    prompt = (
        f"根據這些情緒標籤：{', '.join(emotions)}，請用一段優美、詩意的【中文】文字描述圖片的情緒與氛圍。"
        f"請避免任何英文詞彙，也不要中英夾雜。控制在 80~100 字以內，語氣溫柔、文藝，不要太誇張。"
    )
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2:1b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.4
        }
    })
    return r.json().get("response", "")

def generate_spotify_queries(emotions):
    prompt = f"根據這些情緒：{', '.join(emotions)}，請產生3個適合在 Spotify 搜尋的音樂關鍵字或短語（用英文）。"
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2:1b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.4
        }
    })
    lines = r.json().get("response", "").split('\n')
    return [line.strip('- ') for line in lines if line.strip()]

# Spotify 音樂搜尋
def get_tracks_by_queries(queries, token, per_query=2):
    headers = {"Authorization": f"Bearer {token}"}
    tracks = []
    seen = set()
    for query in queries:
        r = requests.get("https://api.spotify.com/v1/search",
                         headers=headers,
                         params={"q": query, "type": "track", "limit": per_query})
        if r.status_code == 200:
            for item in r.json().get("tracks", {}).get("items", []):
                tid = item["id"]
                if tid not in seen:
                    seen.add(tid)
                    tracks.append({
                        "name": item["name"],
                        "artist": item["artists"][0]["name"],
                        "url": item["external_urls"]["spotify"],
                        "image": item["album"]["images"][0]["url"]
                    })
    return tracks

# 頁面
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/music", methods=["GET", "POST"])
def music():
    if request.method == "POST":
        file = request.files.get("image")
        if file:
            filename = secure_filename(file.filename)
            image = Image.open(file.stream).convert("RGB")
            image.thumbnail((512, 512))
            try:
                emotions = get_top_emotions(image)
                description = generate_emotion_description(emotions)
                queries = generate_spotify_queries(emotions)
                token = get_spotify_token()
                tracks = get_tracks_by_queries(queries, token)
                return render_template("result.html", emotions=emotions, description=description, tracks=tracks)
            except Exception as e:
                return f"發生錯誤：{str(e)}"
    return render_template("music.html")

if __name__ == "__main__":
    app.run(debug=True)

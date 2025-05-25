# ✅ main.py（使用 OpenAI Vision 分析圖片並推薦音樂）
from flask import Flask, request, render_template
from PIL import Image
import requests
import os
import base64
from io import BytesIO
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

# 環境變數
load_dotenv()
app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# Spotify 權杖
def get_spotify_token():
    r = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
    )
    return r.json().get("access_token")

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

# 呼叫 OpenAI Vision 分析圖片並產生情緒與推薦字詞
def analyze_image_with_openai(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {"role": "system", "content": "你是一位藝術與音樂顧問，根據圖片給出情緒描述與適合的 Spotify 音樂搜尋關鍵字。"},
            {"role": "user", "content": [
                {"type": "text", "text": "請描述這張圖片傳遞的情緒（中文），並產生 3 個適合在 Spotify 搜尋的音樂關鍵字（英文）"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
            ]}
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions",
                             headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return "圖片分析失敗。"

# 擷取推薦字詞
import re
def extract_keywords_and_description(text):
    lines = text.strip().split("\n")
    description = lines[0]
    keywords = re.findall(r'"(.*?)"|\b[a-zA-Z0-9\- ]+\b', " ".join(lines[1:]))
    filtered = [kw.strip() for kw in keywords if kw.strip() and len(kw.strip().split()) <= 4]
    return description, filtered[:3]

# 頁面邏輯
@app.route("/", methods=["GET", "HEAD"])
def home():
    return render_template("index.html")

def music():
    if request.method == "POST":
        file = request.files.get("image")
        if file:
            filename = secure_filename(file.filename)
            image = Image.open(file.stream).convert("RGB")
            image.thumbnail((512, 512))
            try:
                raw_text = analyze_image_with_openai(image)
                description, queries = extract_keywords_and_description(raw_text)
                token = get_spotify_token()
                tracks = get_tracks_by_queries(queries, token)
                return render_template("result.html", description=description, tracks=tracks)
            except Exception as e:
                return f"發生錯誤：{str(e)}"
    return render_template("music.html")

if __name__ == "__main__":
    app.run(debug=True)

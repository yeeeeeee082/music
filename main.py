# ✅ main.py（OpenAI API 精簡版）
from flask import Flask, request, render_template
from PIL import Image
import openai
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import requests
import base64
from io import BytesIO

# 環境變數
load_dotenv()
app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# ✅ 把圖片轉成 base64 給 OpenAI 分析

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# ✅ 圖片情緒描述 + 標籤（用 GPT）

def analyze_image_emotion(image):
    base64_image = image_to_base64(image)
    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=[
            {"role": "system", "content": "你是一位具備藝術素養的詩意撰稿人。請描述圖片帶來的情緒與氛圍，並提供 3 個英文情緒標籤給音樂推薦使用。"},
            {"role": "user", "content": [
                {"type": "text", "text": "請幫我分析這張圖片的情緒並產生音樂標籤"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]}
        ],
        max_tokens=300
    )
    text = response.choices[0].message.content.strip()
    # 簡單拆出說明與標籤（以冒號分隔為準）
    parts = text.split("標籤：")
    description = parts[0].strip()
    keywords = parts[1].split(',') if len(parts) > 1 else []
    return description, [kw.strip() for kw in keywords]

# ✅ Spotify 權杖

def get_spotify_token():
    r = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
    )
    return r.json().get("access_token")

# ✅ Spotify 音樂搜尋

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

# ✅ 頁面
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
                description, queries = analyze_image_emotion(image)
                token = get_spotify_token()
                tracks = get_tracks_by_queries(queries, token)
                return render_template("result.html", description=description, tracks=tracks)
            except Exception as e:
                return f"發生錯誤：{str(e)}"
    return render_template("music.html")

if __name__ == "__main__":
    app.run(debug=True)

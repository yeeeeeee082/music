# ✅ streamlit_app.py：使用 Ollama LLaMA3 + 使用者輸入文字（而非圖片）作為分析來源

import streamlit as st
import requests
import base64
from gtts import gTTS
from tempfile import NamedTemporaryFile
import ollama
import os
import re
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

st.set_page_config(page_title="AI 文字情緒分析與音樂推薦", layout="centered")
st.title("📝 文字情緒分析與音樂推薦（LLaMA3）")

text_input = st.text_area("請輸入描述一個場景、情境或心情的文字（中文即可）")

if st.button("分析並推薦音樂"):
    if text_input:
        st.markdown("⏳ **LLaMA3 正在分析中，請稍候...**")

        prompt = (
            f"根據以下的中文描述：{text_input}\n"
            f"請分析它所傳遞的情緒（用中文寫一小段詩意描述），並提供三個適合 Spotify 搜尋的英文音樂關鍵字。\n"
            f"格式如下：\n情緒描述：...\n音樂關鍵字：..."
        )

        response = ollama.chat(
            model="llama3.2:1b",
            messages=[{"role": "user", "content": prompt}]
        )
        result_text = response['message']['content']

        st.subheader("🎨 情緒分析結果")
        st.info(result_text)

        # 語音 TTS
        tts = gTTS(result_text, lang='zh-TW')
        with NamedTemporaryFile(suffix=".mp3", delete=False) as temp:
            tts.save(temp.name)
            audio_bytes = open(temp.name, 'rb').read()
            b64 = base64.b64encode(audio_bytes).decode()
            audio_html = f'<audio controls autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
            st.markdown(audio_html, unsafe_allow_html=True)

        # 提取音樂關鍵字
        keywords = re.findall(r'\b[a-zA-Z0-9\- ]{2,}\b', result_text)
        queries = [kw.strip() for kw in keywords if kw.strip() and len(kw.strip().split()) <= 4][:3]

        # Spotify 音樂推薦
        token = requests.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
        ).json().get("access_token")

        headers = {"Authorization": f"Bearer {token}"}
        tracks = []
        seen = set()
        for query in queries:
            r = requests.get("https://api.spotify.com/v1/search",
                             headers=headers,
                             params={"q": query, "type": "track", "limit": 2})
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

        st.subheader("🎧 推薦音樂")
        for track in tracks:
            st.markdown(f"**{track['name']}** - {track['artist']}")
            st.image(track['image'], width=150)
            st.markdown(f"[Spotify 連結]({track['url']})")

    else:
        st.warning("請輸入文字後再送出分析！")

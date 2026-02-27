import streamlit as st
import requests
import os

API_URL = "http://127.0.0.1:8000"

st.title("VisionArchive AI - Video Intelligence")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])

if uploaded_file:
    if st.button("Index Video"):
        files = {"file": uploaded_file}
        response = requests.post(f"{API_URL}/index-video", files=files)
        st.write("Status Code:", response.status_code)
        st.write("Raw Response:", response.text)

st.divider()

query = st.text_input("Search Videos")

if st.button("Search"):
    response = requests.post(f"{API_URL}/search", params={"query": query})
    
    if response.status_code == 200:
        results = response.json()["results"]

        if not results:
            st.info("No videos found for this query.")
        
        for path, label in results:
            # EXTRACT FILENAME: Convert 'uploaded_videos/name.mp4' to 'name.mp4'
            filename = os.path.basename(path)
            
            # CONSTRUCT URL: Point to the FastAPI static route we just created
            video_url = f"{API_URL}/stream/{filename}"
            
            st.video(video_url)
            st.write(f"**Classification:** {label}")
    else:
        st.error("Error connecting to the backend.")
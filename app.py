import streamlit as st
import requests

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
    results = response.json()["results"]

    for path, label in results:
        st.video(path)
        st.write("Label:", label)

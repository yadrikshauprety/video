import streamlit as st
import requests
import os

API_URL = "http://127.0.0.1:8000"

# Page Configuration for a wider gallery look
st.set_page_config(layout="wide", page_title="VisionArchive AI")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("VisionArchive")
menu = st.sidebar.radio("Navigation", ["Upload & Search", "Video Gallery"])

if menu == "Upload & Search":
    st.title("🔍 Intelligence Search")
    
    # Upload Section
    with st.expander("Upload New Video"):
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi"])
        if uploaded_file and st.button("Index Video"):
            with st.spinner("Processing video..."):
                files = {"file": uploaded_file}
                response = requests.post(f"{API_URL}/index-video", files=files)
                if response.status_code == 200:
                    st.success(f"Indexed! Label: {response.json()['label']}")
                else:
                    st.error("Upload failed.")

    st.divider()

    # Search Section
    query = st.text_input("Describe what you are looking for (e.g., 'someone running')")
    if st.button("Search"):
        response = requests.post(f"{API_URL}/search", params={"query": query})
        if response.status_code == 200:
            results = response.json()["results"]
            if not results:
                st.info("No matching videos found.")
            else:
                # Display results in a grid
                cols = st.columns(2)
                for i, (path, label) in enumerate(results):
                    filename = os.path.basename(path)
                    video_url = f"{API_URL}/stream/{filename}"
                    with cols[i % 2]:
                        st.video(video_url)
                        st.caption(f"Match: {label}")
        else:
            st.error("Search service unavailable.")

elif menu == "Video Gallery":
    st.title("🖼️ Video Gallery")
    
    response = requests.get(f"{API_URL}/videos")
    if response.status_code == 200:
        all_videos = response.json()["videos"]
        
        if not all_videos:
            st.info("Your gallery is empty. Upload some videos to get started!")
        else:
            # Create a 3-column grid for the gallery
            cols = st.columns(3)
            for i, (path, label) in enumerate(all_videos):
                filename = os.path.basename(path)
                video_url = f"{API_URL}/stream/{filename}"
                with cols[i % 3]:
                    st.video(video_url)
                    st.write(f"**Tag:** {label}")
                    st.divider()
    else:
        st.error("Could not load gallery.")
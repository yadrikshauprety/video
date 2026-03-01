import streamlit as st
import requests, os

API_URL = "http://127.0.0.1:8000"
st.set_page_config(layout="wide", page_title="VisionArchive AI Pro")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("VisionArchive")
menu = st.sidebar.radio("Navigation", ["Upload & Search", "Video Gallery", "Identity Manager"])
st.sidebar.divider()
conf_level = st.sidebar.slider("Search Match Confidence", 0.0, 1.0, 0.25, 0.05)

if menu == "Upload & Search":
    st.title("🔍 Semantic Intelligence Search")
    with st.expander("Upload New Video"):
        uploaded_file = st.file_uploader("Choose video...", type=["mp4"])
        if uploaded_file and st.button("Index Video"):
            with st.spinner("Analyzing scenes, actions, and identities..."):
                res = requests.post(f"{API_URL}/index-video", files={"file": uploaded_file})
                if res.status_code == 200:
                    st.success(f"Indexed! Top Action: {res.json()['label']}")

    st.divider()
    query = st.text_input("What activity are you looking for?")
    if st.button("Search"):
        res = requests.post(f"{API_URL}/search", params={"query": query, "threshold": conf_level})
        if res.status_code == 200:
            results = res.json()["results"]
            if not results: st.warning("No high-confidence matches found.")
            else:
                cols = st.columns(2)
                for i, item in enumerate(results):
                    with cols[i % 2]:
                        st.video(f"{API_URL}/stream/{os.path.basename(item['path'])}")
                        st.write(f"**Action:** {item['label']} | **Score:** {item['score']:.2f}")

elif menu == "Identity Manager":
    st.title("👤 Person & Identity Directory")
    res = requests.get(f"{API_URL}/all-persons")
    if res.status_code == 200:
        persons = res.json()["persons"]
        if not persons: st.info("No identities found yet. Index a video to start.")
        else:
            cols = st.columns(5)
            for i, p in enumerate(persons):
                with cols[i % 5]:
                    st.image(f"{API_URL}/faces/{p['thumbnail']}", use_container_width=True)
                    name = p['name'] if p['name'] else f"Unknown #{p['id']}"
                    if st.button(name, key=f"person_btn_{p['id']}"):
                        st.session_state['active_p'] = p

    if 'active_p' in st.session_state:
        p = st.session_state['active_p']
        st.divider()
        col_id, col_vids = st.columns([1, 2])
        
        with col_id:
            st.subheader("Edit Identity")
            new_name = st.text_input("Name this person", value=p['name'] if p['name'] else "")
            if st.button("Save Name"):
                requests.post(f"{API_URL}/name-person/{p['id']}", params={"name": new_name})
                st.success("Name updated!")
                st.rerun()
        
        with col_vids:
            st.subheader(f"Videos featuring this person")
            v_res = requests.get(f"{API_URL}/person-videos/{p['id']}")
            if v_res.status_code == 200:
                for v in v_res.json()["videos"]:
                    st.video(f"{API_URL}/stream/{os.path.basename(v['path'])}")
                    st.caption(f"Activity detected: {v['label']}")

elif menu == "Video Gallery":
    st.title("🖼️ Library Gallery")
    res = requests.get(f"{API_URL}/videos")
    if res.status_code == 200:
        videos = res.json()["videos"]
        cols = st.columns(3)
        for i, v in enumerate(videos):
            with cols[i % 3]:
                st.video(f"{API_URL}/stream/{os.path.basename(v['path'])}")
                st.write(f"**Classification:** {v['label']}")
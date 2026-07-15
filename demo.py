from huggingface_hub import snapshot_download
from pathlib import Path
import streamlit as st
import json

def dmeo():
    local_data = Path(__file__).resolve().parents[0] / "data"
    
    snapshot_download(
        repo_id="Carson-Shively/fluency-trainer",
        repo_type="dataset",
        local_dir=local_data,
        allow_patterns=["sentences/**"]
    )
    
    with open(local_data / "sentences/level1.json", "r") as con:
        level1 = json.load(con)
    with open(local_data / "sentences/level2.json", "r") as con:
        level2 = json.load(con)
    with open(local_data / "sentences/level3.json", "r") as con:
        level3 = json.load(con)
    with open(local_data / "sentences/level4.json", "r") as con:
        level4 = json.load(con)
    with open(local_data / "sentences/level5.json", "r") as con:
        level5 = json.load(con)
    with open(local_data / "sentences/level6.json", "r") as con:
        level6 = json.load(con)
    with open(local_data / "sentences/level7.json", "r") as con:
        level7 = json.load(con)
    with open(local_data / "sentences/level8.json", "r") as con:
        level8 = json.load(con)
    with open(local_data / "sentences/level9.json", "r") as con:
        level9 = json.load(con)
    with open(local_data / "sentences/level10.json", "r") as con:
        level10 = json.load(con)
    with open(local_data / "sentences/level11.json", "r") as con:
        level11 = json.load(con)
    with open(local_data / "sentences/level12.json", "r") as con:
        level12 = json.load(con)
    
    st.slider(
        "Select Level",
        min_value=1,
        max_value=12,
        step=1,
        key="max_level"
    )
    
    st.session_state.current_level = st.session_state.max_level

    
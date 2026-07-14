from huggingface_hub import snapshot_download
from pathlib import Path
import streamlit as st

def dmeo():
    local_data = Path(__file__).resolve().parents[0] / "data"
    
    snapshot_download(
        repo_id="Carson-Shively/fluency-trainer",
        repo_type="dataset",
        local_dir=local_data,
        allow_patterns=["sentences/**"]
    )
    
    st.session_state.max_level = 
    st.session_state.current_level = st.session_state.max_level
    st.session_state.score = 0
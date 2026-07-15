import torch
from transformers import AutoProcessor, AutoModelForCTC
from huggingface_hub import snapshot_download
from pathlib import Path
import streamlit as st
import json
import random
import tensorflow as tf
from fluency_trainer.phone_audio_alignment_model import PhoneAudioAlignmentModel
import soundfile as sf
import numpy as np


@st.cache_data
def load_local_data():
    local_data = Path(snapshot_download(
        repo_id="Carson-Shively/fluency-trainer",
        repo_type="dataset",
        allow_patterns=["sentences/**", "vocab/**"]
    ))
    return local_data
    
@st.cache_data
def load_local_model():
    local_model = Path(snapshot_download(
        repo_id="Carson-Shively/fluency-trainer",
        repo_type="model",
        allow_patterns="phone_audio_alignment_model.weights.h5"
    ))
    return local_model

@st.cache_data
def load_sentences(local_data):
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
        
    levels = {
        1: level1,
        2: level2,
        3: level3,
        4: level4,
        5: level5,
        6: level6,
        7: level7,
        8: level8,
        9: level9,
        10: level10,
        11: level11,
        12: level12
    }
        
    return level1, level2, level3, level4, level5, level6, level7, level8, level9, level10, level11, level12, levels

@st.cache_data
def load_vocab(local_data):
    with open(local_data / "vocab/vocab.json", "r") as con:
        vocab = json.load(con)
    return vocab

@st.cache_resource
def load_model(local_model, vocab_len):
    model = PhoneAudioAlignmentModel(vocab_size=vocab_len)
    model(audio=tf.zeros((1, 1, 392), dtype=tf.float32), phones=tf.zeros((1, 1), dtype=tf.int32), audio_mask=tf.zeros((1, 1), dtype=tf.float32), phones_mask=tf.zeros((1, 1), dtype=tf.float32))
    model.load_weights(local_model / "phone_audio_alignment_model.weights.h5")

    return model

@st.cache_resource
def load_wav2vec2():
    wav2vec2_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    wav2vec2_model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    wav2vec2_model.eval()
    return wav2vec2_processor, wav2vec2_model

def demo():

    local_data = load_local_data()
    local_model = load_local_model()
    level1, level2, level3, level4, level5, level6, level7, level8, level9, level10, level11, level12, levels = load_sentences(local_data)
    vocab = load_vocab(local_data)
    model = load_model(local_model, len(vocab))
    wav2vec2_processor, wav2vec2_model = load_wav2vec2()

    
    st.slider(
        "Select Level",
        min_value=1,
        max_value=12,
        step=1,
        key="selected_level"
    )
    

        
    
    if "current_sentence" not in st.session_state:
        st.session_state.current_sentence = random.choice(levels[st.session_state.selected_level]["sentences"])
        st.session_state.current_level = st.session_state.selected_level
    elif st.session_state.current_level != st.session_state.selected_level:
        st.session_state.current_level = st.session_state.selected_level
        st.session_state.current_sentence = random.choice(levels[st.session_state.selected_level]["sentences"])
    
    if st.button("New Sentence"):
        st.session_state.current_sentence = random.choice(levels[st.session_state.selected_level]["sentences"])

    st.write(st.session_state.current_sentence["sentence"])
    
    phones = []
    
    for word in st.session_state.current_sentence["words"]:
        phones.extend(word["phonemes"])
        
    phone_ids = []
        
    for phone in phones:
        if phone in vocab:
            phone_ids.append(vocab[phone])
        else:
            phone_ids.append(vocab["<UNK>"])
            
    phone_ids_tensor = tf.convert_to_tensor([phone_ids], dtype=tf.int32)
    
    phones_mask_tensor = tf.ones(shape=tf.shape(phone_ids_tensor), dtype=tf.float32)
    
    user_audio = st.audio_input("Record", sample_rate=16000)
    
    if user_audio == None:
        return
    
    waveform, sample_rate = sf.read(user_audio)
    
    padding = int(16000 * 0.1)
    
    active_indexes = np.where(np.abs(waveform) > 0.01)[0]
    
    if len(active_indexes) > 0:
        start_index = max((active_indexes[0] - padding), 0)
        end_index = min((active_indexes[-1] + padding) + 1, len(waveform))
        
        waveform = waveform[start_index:end_index]
    
    
    processed_user_audio = wav2vec2_processor(
        waveform,
        sampling_rate=sample_rate,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        wav2vec2_out = wav2vec2_model(**processed_user_audio)
    
    user_audio_logits = wav2vec2_out.logits
    user_audio_np = user_audio_logits.numpy()
    user_audio_tensor = tf.convert_to_tensor(user_audio_np, dtype=tf.float32)
    user_audio_mask_tensor = tf.ones(shape=tf.shape(user_audio_tensor)[:2], dtype=tf.float32)
    
    score = model(audio=user_audio_tensor, phones=phone_ids_tensor, audio_mask=user_audio_mask_tensor, phones_mask=phones_mask_tensor)
    
    score_total = score["total"][0, 0]
    score_total_value = float(score_total.numpy())
    output_score_total = round((score_total_value * 10.0), 2)
    
    st.write(f"score: {output_score_total}%")

    
    
    

    
if __name__ == "__main__":
    demo()
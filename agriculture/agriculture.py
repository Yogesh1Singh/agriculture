import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
from gtts import gTTS
import os

# Simple cloud detection based on colors (later you can replace with ML model)
def detect_cloud_type(image):
    img = np.array(image)
    avg_color = img.mean(axis=(0, 1))  # [B, G, R]

    if avg_color[2] > avg_color[1] and avg_color[2] > avg_color[0]:
        return "Cirrus (Thin, White Clouds)"
    elif avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2]:
        return "Cumulus (Fluffy White Clouds)"
    else:
        return "Stratus (Grayish Layered Clouds)"

# Simple moisture detection from nostoc color
def detect_moisture_level(image):
    img = np.array(image)

    avg_color = img.mean(axis=(0, 1))  # [B, G, R]

    if avg_color[1] > avg_color[2] and avg_color[1] > avg_color[0]:
        return "Moist (Bright Green/Blue-Green)"
    else:
        return "Dry (Dark Green/Brown/Black)"

# Combined advice based on moisture & cloud
def generate_combined_advice(moisture, cloud_type):
    advice = ""

    if moisture == "Moist (Bright Green/Blue-Green)":
        advice += "मिट्टी में नमी अच्छी है। उर्वरक की मात्रा कम रखें।\n"
    else:
        advice += "खेत सूखा है, तुरंत सिंचाई करें। 50 लीटर पानी प्रति बीघा डालें।\n"

    if cloud_type in ["Cirrus (Thin, White Clouds)", "Cirrocumulus", "Cirrostratus"]:
        advice += "⚠️ चेतावनी: आसमान में चक्रवात के लक्षण दिख रहे हैं। खेत को सुरक्षित करें।\n"
    else:
        advice += "मौसम सामान्य है, कीटनाशक और उर्वरक का संतुलन बनाएं।\n"

    advice += f"(Moisture: {moisture}, Cloud Type: {cloud_type})"
    return advice

def text_to_speech(text, lang='hi'):
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        tts.save(temp_audio.name)
        return temp_audio.name

st.title("🌾 AI Krishi Assistant - Moisture & Cyclone Predictor")

st.header("Step 1: खेत की मिट्टी (Nostoc) की फोटो खींचें")
soil_image = st.camera_input("कृपया Nostoc वाली मिट्टी की फोटो लें")

st.header("Step 2: आकाश की फोटो खींचें")
sky_image = st.camera_input("कृपया आकाश की फोटो लें")

if soil_image and sky_image:
    # Process Soil (Moisture Detection)
    soil_img = Image.open(soil_image)
    st.image(soil_img, caption="Nostoc Image (Soil Condition)", use_column_width=True)
    moisture_level = detect_moisture_level(soil_img)

    # Process Sky (Cloud Detection)
    sky_img = Image.open(sky_image)
    st.image(sky_img, caption="Sky Image (Cloud Condition)", use_column_width=True)
    cloud_type = detect_cloud_type(sky_img)

    st.write(f"📊 Moisture Level: **{moisture_level}**")
    st.write(f"☁️ Cloud Type: **{cloud_type}**")

    advice = generate_combined_advice(moisture_level, cloud_type)
    st.write("🧑‍🌾 **किसान सलाह:**")
    st.write(advice)

    # Hindi Audio Advice
    audio_file = text_to_speech(advice)

    st.audio(audio_file, format='audio/mp3')

    # Clean temp audio file
    os.remove(audio_file)

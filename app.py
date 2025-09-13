import streamlit as st
from langdetect import detect, LangDetectException
import google.generativeai as genai
from dotenv import load_dotenv
import os
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from io import BytesIO
import json
import requests
import pandas as pd

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-pro')
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

texts = {
    'ask': {'en':'Ask your question:', 'hi':'अपना प्रश्न पूछें:', 'ml':'നിങ്ങളുടെ ചോദ്യം ചോദിക്കുക:'},
    'samples': {'en':'Try these sample questions:', 'hi':'इन उदाहरण प्रश्नों को आज़माएँ:', 'ml':'ഈ ഉദാഹരണ ചോദ്യങ്ങൾ ശ്രമിക്കുക:'},
    'record': {'en':'🎤 Record Question', 'hi':'🎤 प्रश्न बोलें', 'ml':'🎤 ചോദ്യം റെക്കോർഡ് ചെയ്യുക'},
    'name': {'en': 'Name', 'hi': 'नाम', 'ml': 'പേര്'},
    'location': {'en': 'Location (e.g., Jabalpur, India)', 'hi': 'स्थान (जैसे, जबलपुर, भारत)', 'ml': 'സ്ഥലം (ഉദാഹരണം, ജബൽപൂർ, ഇന്ത്യ)'},
    'crop': {'en': 'Main Crop (e.g., Wheat, Rice)', 'hi': 'मुख्य फसल (जैसे, गेहूं, चावल)', 'ml': 'പ്രധാന വിള (ഉദാഹരണം, ഗോതമ്പ്, അരി)'},
    'password': {'en': 'Password', 'hi': 'पासवर्ड', 'ml': 'പാസ്വേഡ്'},
    'register_button': {'en': 'Register', 'hi': 'पंजीकरण करें', 'ml': 'രജിസ്റ്റർ ചെയ്യുക'},
    'login_button': {'en': 'Login', 'hi': 'लॉग इन करें', 'ml': 'ലോഗിൻ ചെയ്യുക'}
}
sample_qs = {
    'en': ["When should I irrigate my crop?", "Is there rain tomorrow?"],
    'hi': ["क्या मुझे कल कीटनाशक छिड़काव करना चाहिए?", "मेरे जिले में मौसम कैसा है?"],
    'ml': ["എന്റെ കൃഷിക്ക് എപ്പോഴാണ് ജലസേചനം?", "നാളെ മഴ പെയ്യുമോ?"]
}

def label(t, lang):
    return t.get(lang, t['en'])

def load_db():
    if not os.path.exists('farmers_db.json'):
        with open('farmers_db.json', 'w') as f:
            json.dump([], f)
    with open('farmers_db.json', 'r') as f:
        return json.load(f)

def save_db(data):
    with open('farmers_db.json', 'w') as f:
        json.dump(data, f, indent=2)

def get_coords(location_name):
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={location_name}&limit=1&appid={OPENWEATHERMAP_API_KEY}"
    try:
        response = requests.get(url).json()
        if response:
            return response[0]['lat'], response[0]['lon']
    except Exception:
        return None, None
    return None, None

def get_weather(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHERMAP_API_KEY}&units=metric"
    try:
        data = requests.get(url).json()
        description = data['weather'][0]['description'].title()
        temp = data['main']['temp']
        return f"{temp}°C, {description}"
    except Exception:
        return "Weather data unavailable."

if 'lang' not in st.session_state:
    st.session_state.lang = 'en'
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.selected_farmer = None

st.title("🌱 Krishi Sakhi – AI Farming Assistant")

if not st.session_state.logged_in:
    login_tab, register_tab = st.sidebar.tabs(["Login", "Register"])
    with login_tab:
        st.subheader("Login to your Profile")
        with st.form("login_form"):
            login_name = st.text_input(label(texts['name'], st.session_state.lang))
            login_password = st.text_input(label(texts['password'], st.session_state.lang), type="password")
            login_submitted = st.form_submit_button(label(texts['login_button'], st.session_state.lang))
            if login_submitted:
                farmers = load_db()
                found_farmer = next((f for f in farmers if f['name'] == login_name), None)
                if found_farmer and found_farmer['password'] == login_password:
                    st.session_state.logged_in = True
                    st.session_state.selected_farmer = found_farmer
                    st.rerun()
                else:
                    st.error("Invalid name or password.")
    with register_tab:
        st.subheader("Create a New Profile")
        with st.form("registration_form"):
            new_name = st.text_input(label(texts['name'], st.session_state.lang))
            new_password = st.text_input(label(texts['password'], st.session_state.lang), type="password")
            new_location = st.text_input(label(texts['location'], st.session_state.lang))
            new_crop = st.text_input(label(texts['crop'], st.session_state.lang))
            register_submitted = st.form_submit_button(label(texts['register_button'], st.session_state.lang))
            if register_submitted:
                farmers = load_db()
                if any(f['name'] == new_name for f in farmers):
                    st.error("A farmer with this name already exists. Please choose another name.")
                elif new_name and new_location and new_crop and new_password:
                    lat, lon = get_coords(new_location)
                    if lat and lon:
                        farmers.append({"name": new_name, "password": new_password, "location": new_location, "crop": new_crop, "lat": lat, "lon": lon})
                        save_db(farmers)
                        st.success(f"Farmer {new_name} registered! Please login.")
                    else:
                        st.error("Could not find location. Please be more specific.")
                else:
                    st.error("Please fill all fields.")
    st.info("Please Login or Register using the sidebar to continue.")
else:
    farmer = st.session_state.selected_farmer
    st.sidebar.success(f"Logged in as {farmer['name']}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.selected_farmer = None
        st.rerun()

    st.subheader(f"Dashboard for {farmer['name']}")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric(label="📍 Location", value=farmer['location'])
        st.metric(label="🌾 Main Crop", value=farmer['crop'])
        st.markdown("##### 🌦️ Current Weather")
        weather_info = get_weather(farmer['lat'], farmer['lon'])
        st.write(weather_info)
    with col2:
        df = pd.DataFrame({'lat': [farmer['lat']], 'lon': [farmer['lon']]})
        st.map(df, zoom=10)

    with st.expander("💡 Tip of the Day"):
        with st.spinner("Generating today's tip..."):
            tip_prompt = f"Based on this info: farmer's location is {farmer['location']}, main crop is {farmer['crop']}, and the weather is '{weather_info}'. Provide one short, actionable farming tip for today."
            tip = model.generate_content(tip_prompt).text
            st.info(tip)
    st.markdown("---")

    user_input = st.text_input(label(texts['ask'], st.session_state.lang), key="typed")
    try:
        if user_input: st.session_state.lang = detect(user_input)
    except LangDetectException:
        st.session_state.lang = 'en'

    st.markdown(f"**{label(texts['samples'], st.session_state.lang)}**")
    scol1, scol2, scol3 = st.columns(3)
    with scol1:
        st.markdown("**In English**")
        for q in sample_qs['en']:
            if st.button(q, use_container_width=True):
                user_input = q
                st.session_state.lang = 'en'
    with scol2:
        st.markdown("**हिंदी में**")
        for q in sample_qs['hi']:
            if st.button(q, use_container_width=True):
                user_input = q
                st.session_state.lang = 'hi'
    with scol3:
        st.markdown("**മലയാളത്തിൽ**")
        for q in sample_qs['ml']:
            if st.button(q, use_container_width=True):
                user_input = q
                st.session_state.lang = 'ml'
    
    if st.button(label(texts['record'], st.session_state.lang)):
        lang_code_map = {'en': 'en-IN', 'hi': 'hi-IN', 'ml': 'ml-IN'}
        recognition_lang = lang_code_map.get(st.session_state.lang, 'en-IN')
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening...")
            audio = r.listen(source, phrase_time_limit=5)
        try:
            voice_text = r.recognize_google(audio, language=recognition_lang)
            st.success(f"You said: {voice_text}")
            user_input = voice_text
            try: st.session_state.lang = detect(user_input)
            except LangDetectException: st.session_state.lang = 'en'
        except Exception as e:
            st.error("Could not recognize speech")

    if user_input:
        contextual_prompt = f"""
        You are an expert farming assistant. A farmer is asking a question. Here is the farmer's context:
        - Name: {farmer['name']}
        - Location: {farmer['location']}
        - Main Crop: {farmer['crop']}
        - Current Weather: {weather_info}
        Based on all this context, answer the following question: "{user_input}"
        """
        with st.spinner("Analyzing your question with local data..."):
            response = model.generate_content(contextual_prompt).text
            st.subheader("🤖 Here is your personalized advice:")
            st.write(response)
            try:
                tts = gTTS(text=response, lang=st.session_state.lang)
            except Exception:
                tts = gTTS(text=response, lang='en')
            audio_bytes = BytesIO()
            tts.write_to_fp(audio_bytes)
            st.audio(audio_bytes, format="audio/mp3", start_time=0)
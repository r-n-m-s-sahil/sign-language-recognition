import streamlit as st
import cv2
import mediapipe as mp
import pickle
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.stApp {
    background-color: #0E1117;
    color: white;
}

.title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: #00FFAA;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #AAAAAA;
    margin-bottom: 20px;
}

.prediction-box {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    color: #00FFAA;
    border: 2px solid #00FFAA;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown(
    '<p class="title">🤟 Sign Language Recognition System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Real-time Hand Gesture Recognition using MediaPipe & Machine Learning</p>',
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 Project Information")

st.sidebar.info("""
This project uses:

- OpenCV
- MediaPipe
- Machine Learning
- Streamlit

to detect and recognize sign language gestures in real-time.
""")

st.sidebar.success("Model Status: Loaded")

st.sidebar.markdown("---")

st.sidebar.markdown("## Instructions")

st.sidebar.write("""
1. Click Start Camera  
2. Show hand gesture  
3. Prediction appears live  
4. Press Stop to end  
""")

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# ---------------- CAMERA BUTTON ----------------
run = st.checkbox("✅ Start Camera")

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([3,1])

with col1:
    FRAME_WINDOW = st.image([])

with col2:
    prediction_placeholder = st.empty()

# ---------------- WEBCAM ----------------
cap = cv2.VideoCapture(0)

while run:

    ret, frame = cap.read()

    if not ret:
        st.error("Failed to access webcam")
        break

    # Flip frame
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hands
    results = hands.process(rgb)

    prediction = "No Hand"

    confidence = 0

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract landmarks
            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Prediction
            probs = model.predict_proba([landmarks])[0]

            class_index = np.argmax(probs)

            confidence = probs[class_index] * 100

            prediction = model.classes_[class_index]

    # Draw prediction on frame
    cv2.putText(
        frame,
        f"{prediction}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0,255,0),
        3
    )

    # Convert image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Show webcam
    FRAME_WINDOW.image(frame)

    # Show prediction box
    prediction_placeholder.markdown(f"""
    <div class="prediction-box">
        Prediction<br><br>
        {prediction}
        <br><br>
        Confidence: {confidence:.2f}%
    </div>
    """, unsafe_allow_html=True)

cap.release()
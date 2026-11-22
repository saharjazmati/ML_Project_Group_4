
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import base64

from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Ottawa Vehicle Theft & Recovery",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load background image
app_folder = Path(__file__).parent
image_path = app_folder / "background.jpeg"

background = base64.b64encode(
    image_path.read_bytes()
).decode()

# Custom visual design
st.markdown(f"""
<style>

.stApp {{
    background-image:
        linear-gradient(
            rgba(3, 8, 20, 0.20),
            rgba(3, 8, 20, 0.35)
        ),
        url("data:image/jpeg;base64,{background}");
    background-size: cover;
    background-position: center bottom;
    background-attachment: fixed;
}}

[data-testid="stHeader"] {{
    background: transparent;
}}

[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stToolbar"] {{
    display: none;
}}

.block-container {{
    max-width: 1400px;
    margin-left: auto;
    margin-right: auto;
    padding-top: 3rem;
    padding-bottom: 2rem;
}}

h1, h2, h3, p, label {{
    color: white !important;
}}

[data-testid="stCaptionContainer"] p {{
    color: #ccd6e5 !important;
}}

[data-testid="stForm"] {{
    background: rgba(10, 20, 40, 0.40);
    backdrop-filter: blur(3px);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 24px;
    padding: 20px;
    box-shadow:
        -6px 0 30px rgba(0,100,255,0.15),
        6px 0 30px rgba(255,0,0,0.15);
}}

[data-baseweb="select"] > div {{
    background: rgba(15, 25, 45, 0.95);
    color: white;
    border-radius: 10px;
    border: 1px solid #64748b;
}}

[data-baseweb="select"] span {{
    color: white;
}}

[data-testid="stFormSubmitButton"] button {{
    background: linear-gradient(
        110deg, #1456d9, #7335b5, #d91e36
    );
    color: white;
    border: none;
    border-radius: 12px;
    font-size: 18px;
    font-weight: 700;
    min-height: 55px;
    width: 100%;
    transition: all 0.3s ease;
}}

[data-testid="stFormSubmitButton"] button:hover {{
    box-shadow: 0 0 25px rgba(60,100,255,0.7);
    transform: translateY(-2px);
    color: white;
}}

.result-panel {{
    background: rgba(8,15,30,0.85);
    border-radius: 24px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(3px);
    animation: appear 0.8s ease;
}}

@keyframes appear {{
    from {{
        opacity: 0;
        transform: translateY(15px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}


    @media (max-width: 800px) {{
        .block-container {{
            margin-left: auto;
            margin-right: auto;
            max-width: 100%;
        }}
    }}
    </style>
    
""", unsafe_allow_html=True)

# Load trained models
@st.cache_resource
def load_model():
    return joblib.load(
        app_folder / "soft_voting_model.joblib"
    )

bundle = load_model()

models = bundle["models"]
predictors = bundle["predictors"]
categories = bundle["categories"]
threshold = bundle["threshold"]

left_panel, right_panel = st.columns([1.1, 1], gap='large')

with left_panel:
    # Application title
    st.markdown("""
    <div style="text-align:center; margin-bottom:35px;">
        <div style="font-size:14px; letter-spacing:5px;
                    color:#b8c8e0;">
            OTTAWA · CANADA
        </div>
        <h1 style="font-size:clamp(32px,5vw,48px); font-weight:800;">
            VEHICLE THEFT & RECOVERY
        </h1>
        <div style="font-size:21px; color:#e2e8f0;">
            Vehicle Recovery Prediction
        </div>
        <div style="font-size:13px; color:#a8b5ca;
                    margin-top:15px;">
            Ottawa Police Service · 2018–2025
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Input form
    with st.form("prediction_form"):

        st.subheader("Vehicle and Theft Characteristics")

        col1, col2 = st.columns(2)
        selected_values = {}

        for i, predictor in enumerate(predictors):

            column = col1 if i % 2 == 0 else col2

            with column:
                selected_values[predictor] = st.selectbox(
                    predictor.replace("_", " "),
                    options=categories[predictor]
                )

        submitted = st.form_submit_button(
            "Predict Recovery",
            use_container_width=True
        )


with right_panel:
    # Prediction
    if submitted:

        new_case = pd.DataFrame(
            [selected_values],
            columns=predictors
        )

        probabilities = [
            model.predict_proba(new_case)[0, 1]
            for model in models.values()
        ]

        recovery_probability = float(np.mean(probabilities))
        non_recovery_probability = 1 - recovery_probability

        prediction = (
            "LIKELY TO BE RECOVERED"
            if recovery_probability >= threshold
            else "UNLIKELY TO BE RECOVERED"
        )

        # Result colour
        if recovery_probability > 0.70:
            color = "#22e88a"
            symbol = "🎆"
        elif recovery_probability >= threshold:
            color = "#4ade80"
            symbol = "✓"
        elif recovery_probability >= 0.40:
            color = "#fbbf24"
            symbol = "!"
        else:
            color = "#ff5252"
            symbol = "✕"

        # Animated fireworks above 70%
        if recovery_probability > 0.70:

            components.html("""
            <canvas id="fireworks"></canvas>

            <style>
            body {
                margin: 0;
                overflow: hidden;
                background: transparent;
            }
            canvas {
                width: 100%;
                height: 220px;
            }
            </style>

            <script>
            const canvas = document.getElementById("fireworks");
            const ctx = canvas.getContext("2d");

            canvas.width = window.innerWidth;
            canvas.height = 220;

            let particles = [];

            const colors = [
                "#ff1744", "#2979ff", "#ffea00",
                "#00e676", "#ffffff", "#e040fb"
            ];

            function burst() {

                const x = Math.random() * canvas.width;
                const y = 30 + Math.random() * 120;

                const color = colors[
                    Math.floor(Math.random() * colors.length)
                ];

                for (let i = 0; i < 80; i++) {

                    const angle = Math.random() * Math.PI * 2;
                    const speed = 1 + Math.random() * 5;

                    particles.push({
                        x: x,
                        y: y,
                        vx: Math.cos(angle) * speed,
                        vy: Math.sin(angle) * speed,
                        life: 1,
                        color: color
                    });
                }
            }

            const start = performance.now();

            function animate(now) {

                ctx.clearRect(
                    0, 0, canvas.width, canvas.height
                );

                if (now - start < 4500 &&
                    Math.random() < 0.06) {
                    burst();
                }

                particles = particles.filter(p => p.life > 0);

                particles.forEach(p => {

                    p.x += p.vx;
                    p.y += p.vy;
                    p.vy += 0.035;
                    p.life -= 0.012;

                    ctx.globalAlpha = Math.max(0, p.life);
                    ctx.fillStyle = p.color;

                    ctx.beginPath();
                    ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2);
                    ctx.fill();
                });

                if (now - start < 6500) {
                    requestAnimationFrame(animate);
                } else {
                    ctx.clearRect(
                        0, 0, canvas.width, canvas.height
                    );
                }
            }

            burst();
            burst();
            requestAnimationFrame(animate);

            </script>
            """, height=230)
        else:
            st.html('<div style="height:230px;"></div>')

        # Artistic result panel
        st.html(f"""
        <div class="result-panel"
             style="border:1px solid {color};
                    box-shadow:0 0 35px {color}33;
                    margin-top:25px;">

            <div style="color:#a8b5ca;
                        letter-spacing:3px;
                        font-size:13px;">
                PREDICTED OUTCOME
            </div>

            <div style="font-size:26px;
                        font-weight:800;
                        color:{color};
                        margin:12px;">
                {symbol} {prediction}
            </div>

            <div style="font-size:70px;
                        font-weight:800;
                        color:{color};
                        text-shadow:0 0 25px {color}66;">
                {recovery_probability:.1%}
            </div>

            <div style="color:white; font-size:17px;">
                Recovery Probability
            </div>

            <div style="background:#263247;
                        height:14px;
                        border-radius:20px;
                        margin:25px 0;
                        overflow:hidden;">

                <div style="background:{color};
                            height:100%;
                            width:{recovery_probability * 100}%;
                            border-radius:20px;">
                </div>
            </div>

            <div style="color:#cbd5e1; font-size:18px;">
                Non-Recovery Probability:
                <strong>{non_recovery_probability:.1%}</strong>
            </div>

        </div>
        """)


# Disclaimer
st.markdown("""
<div style="text-align:center;
            color:#a8b5ca;
            font-size:12px;
            margin-top:40px;">
    Educational demonstration only.
    Predictions do not guarantee actual recovery outcomes.
</div>
""", unsafe_allow_html=True)

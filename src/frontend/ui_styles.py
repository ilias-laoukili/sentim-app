import streamlit as st


def apply_styles() -> None:
    st.markdown(
        """
        <style>
            /* --- FONT IMPORT --- */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');

            /* --- GENERAL STYLES --- */
            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif;
            }

            .stApp {
                background: linear-gradient(180deg, rgba(227, 255, 238, 0.2) 0%, #F9FAFB 20%);
            }

            /* --- HIDE STREAMLIT DEFAULTS --- */
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}

            /* --- TYPOGRAPHY --- */
            h1, h2, h3, h4, h5, h6 {
                font-family: 'Plus Jakarta Sans', sans-serif;
                font-weight: 800;
                color: #111827;
                letter-spacing: -1.5px;
            }
            
            p, .stMarkdown {
                color: #4B5563;
                font-size: 1.1rem;
                line-height: 1.6;
            }

            /* --- BUTTONS --- */
            .stButton > button {
                border-radius: 50px !important;
                padding: 16px 32px !important;
                font-weight: 800 !important;
                font-size: 1.1rem !important;
                border: 3px solid #FFFFFF !important;
                transition: all 0.2s ease-in-out !important;
                background-color: #000000 !important;
                color: #FFFFFF !important;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
                text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5) !important;
                letter-spacing: 0.5px !important;
            }
            
            /* Force white text on all button children */
            .stButton > button p,
            .stButton > button span,
            .stButton > button div,
            .stButton > button * {
                color: #FFFFFF !important;
            }

            .stButton > button:hover {
                transform: scale(1.05);
                background-color: #1F2937 !important;
                box-shadow: 0 6px 16px rgba(0, 0, 0, 0.4) !important;
                border: 3px solid #F3F4F6 !important;
            }
            
            .stButton > button:hover p,
            .stButton > button:hover span,
            .stButton > button:hover div,
            .stButton > button:hover * {
                color: #FFFFFF !important;
            }
            
            /* Primary button style */
            button[kind="primary"] {
                background-color: #000000 !important;
                color: #FFFFFF !important;
                font-size: 1.15rem !important;
                font-weight: 800 !important;
                padding: 18px 36px !important;
            }
            
            button[kind="primary"] p,
            button[kind="primary"] span,
            button[kind="primary"] div,
            button[kind="primary"] * {
                color: #FFFFFF !important;
            }

            /* --- SLIDERS --- */
            .stSlider > div > div > div {
                background-color: #000000 !important;
            }
            
            .stSlider > div > div > div > div {
                background-color: #FFFFFF !important;
                border: 2px solid #000000 !important;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
            }
            
            .stSlider > div > div > div > div:hover {
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
            }
            
            /* Remove black background from slider tick labels */
            .stSlider [data-testid="stTickBarMin"],
            .stSlider [data-testid="stTickBarMax"] {
                background-color: transparent !important;
                color: #000000 !important;
                font-weight: 600 !important;
            }
            
            /* Force black text for slider values */
            .stSlider [data-baseweb="tooltip"],
            .stSlider [data-baseweb="slider"] span,
            .stSlider div[role="slider"],
            .stSlider .st-emotion-cache-1gulkj5,
            .stSlider span,
            .stSlider p,
            .stSlider > div span,
            .stSlider > div > div span,
            .stSlider > div > div > div span,
            [class*="StyledThumbValue"],
            [class*="thumbValue"],
            div[data-baseweb="slider"] span {
                color: #000000 !important;
                font-weight: 600 !important;
            }
            
            /* Target all text within slider container */
            .stSlider * {
                color: #000000 !important;
            }
            
            /* Exception: keep emoji/icons their natural color */
            .stSlider .emoji {
                color: inherit !important;
            }

            /* --- CONTAINERS & CARDS --- */
            .custom-container {
                border-radius: 16px;
                border: 1px solid #E5E7EB;
                padding: 2rem;
                background-color: #FFFFFF;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            }

            /* --- FILE UPLOADER --- */
            .stFileUploader {
                border: 2px dashed #E5E7EB;
                border-radius: 16px;
                padding: 2rem;
                background-color: #FFFFFF;
            }
            
            .stFileUploader label {
                font-weight: 600;
                color: #111827;
            }

            /* --- METRICS --- */
            [data-testid="stMetricValue"] {
                color: #111827 !important;
            }
            
            [data-testid="stMetricLabel"] {
                color: #4B5563 !important;
            }
            
            [data-testid="stMetricDelta"] {
                color: #059669 !important;
            }

        </style>
    """,
        unsafe_allow_html=True,
    )

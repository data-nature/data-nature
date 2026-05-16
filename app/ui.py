import streamlit as st

def set_page_config(title="Data Nature", icon="🌿", layout="wide"):
    """
    Sets the Streamlit page configuration and injects global custom CSS.
    This must be the first Streamlit command called on any page.
    """
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout=layout,
        initial_sidebar_state="expanded"
    )
    
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* Global Font */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Modern Gradient Headers */
        h1, h2, h3 {
            background: -webkit-linear-gradient(45deg, #10b981, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }

        /* Sidebar styling (Glassmorphism effect) */
        [data-testid="stSidebar"] {
            background-color: rgba(30, 41, 59, 0.7) !important;
            backdrop-filter: blur(12px) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* Button styling */
        .stButton > button {
            border-radius: 8px;
            transition: all 0.3s ease-in-out;
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            border: none;
            color: white;
            box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.4);
            font-weight: 600;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.5);
            color: white;
            border: none;
        }

        /* Cards/Metrics styling */
        [data-testid="metric-container"] {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        
        [data-testid="metric-container"]:hover {
            transform: translateY(-2px);
        }

        /* Main container padding adjustments */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

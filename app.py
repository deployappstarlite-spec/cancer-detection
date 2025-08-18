import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import hashlib

# Set page configuration
st.set_page_config(
    page_title="Blood Cell AI Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Abstract CSS Theme (same as original)
st.markdown("""
<style>
/* (CSS omitted for brevity – unchanged) */
</style>
""", unsafe_allow_html=True)

# --- Updated cell classification ---
BLOOD_CELL_CLASSES = {
    0: "Malignant (early pre-B) lymphoblast",
    1: "Malignant (pre-B) lymphoblast",
    2: "Malignant (pro-B) lymphoblast",
    3: "Benign"
}

# --- Oncology-accurate dynamic medical information ---
DYNAMIC_CELL_INFO = {
    "Malignant (pro-B) lymphoblast": {
        "base_info": {
            "stage": "Earliest B-lineage (pro-B / early precursor) in B-ALL",
            "presence": "Not present in normal peripheral blood",
            "typical_immunophenotype": [
                "CD19+, CD22+, PAX5+",
                "TdT+, CD34+",
                "CD10− or dim",
                "cμ− (no cytoplasmic μ), no surface Ig"
            ],
            "morphology_features": [
                "High N:C ratio",
                "Fine/loose chromatin",
                "0–2 nucleoli",
                "Scant basophilic cytoplasm"
            ]
        },
        "dynamic_suggestions": [
            {
                "condition": "elevated_confidence",
                "insights": [
                    "Blast-like fine chromatin with nucleolus is consistent with pro-B lymphoblast",
                    "Absent surface immunoglobulin features expected for this stage",
                    "Cytomorphology suggests immature B-lineage; correlate with TdT and CD34",
                    "Consider karyotype/NGS panel for risk stratification"
                ]
            },
            {
                "condition": "moderate_confidence",
                "insights": [
                    "Some features overlap with other blasts; immunophenotyping recommended",
                    "Chromatin fineness partly appreciable; higher resolution image could help",
                    "Assess CD10 to distinguish pro-B from common/pre-B"
                ]
            },
            {
                "condition": "low_confidence",
                "insights": [
                    "Definitive blast features subtle; obtain better focus/contrast",
                    "Request flow cytometry for classification",
                    "Manual expert review advised"
                ]
            }
        ]
    },
    "Malignant (early pre-B) lymphoblast": {
        "base_info": {
            "stage": "Early pre-B lymphoblast (overlaps with pro-B terminology)",
            "presence": "Not present in normal peripheral blood",
            "typical_immunophenotype": [
                "CD19+, CD22+, PAX5+",
                "TdT+, CD34+",
                "CD10+ (CALLA) usually present",
                "cμ− to weak, no surface Ig"
            ],
            "morphology_features": [
                "Round/oval nucleus",
                "Fine chromatin with 1–2 nucleoli",
                "Scant to moderate cytoplasm"
            ]
        },
        "dynamic_suggestions": [
            {
                "condition": "elevated_confidence",
                "insights": [
                    "High N:C ratio with fine chromatin fits early pre-B blast",
                    "CD10 positivity typical at this stage; verify by flow cytometry",
                    "Absence of surface Ig aligns with precursor status"
                ]
            },
            {
                "condition": "moderate_confidence",
                "insights": [
                    "Nucleolar prominence partly visible; improved illumination may help",
                    "Differentiate from pro-B by CD10 expression",
                    "Correlate with cμ status if available"
                ]
            },
            {
                "condition": "low_confidence",
                "insights": [
                    "Morphology insufficient for staging; recommend immunophenotype workup",
                    "Improve focus/contrast for chromatin detail",
                    "Expert morphologic review suggested"
                ]
            }
        ]
    },
    "Malignant (pre-B) lymphoblast": {
        "base_info": {
            "stage": "Pre-B (common B-ALL) — later precursor than pro-/early pre-B",
            "presence": "Not present in normal peripheral blood",
            "typical_immunophenotype": [
                "CD19+, CD22+, PAX5+, CD10+ (CALLA)",
                "TdT+, variable CD34",
                "Cytoplasmic μ chain positive (cμ+)",
                "No surface immunoglobulin"
            ],
            "morphology_features": [
                "Round nucleus with fine chromatin",
                "Visible nucleoli",
                "Moderate cytoplasm (more than earliest blasts)",
                "High N:C ratio, but less extreme than pro-B"
            ]
        },
        "dynamic_suggestions": [
            {
                "condition": "elevated_confidence",
                "insights": [
                    "Features consistent with pre-B blast; correlate with cμ positivity",
                    "CD10 (CALLA) expression typical; confirm immunophenotype",
                    "Consider MRD methods for monitoring if clinical context applies"
                ]
            },
            {
                "condition": "moderate_confidence",
                "insights": [
                    "Cytoplasmic volume suggests pre-B; verify with cμ",
                    "Chromatin/nucleoli partly appreciable; higher magnification recommended",
                    "Rule out aberrant myeloid antigen expression"
                ]
            },
            {
                "condition": "low_confidence",
                "insights": [
                    "Indistinct nuclear detail; refine focus and contrast",
                    "Use flow cytometry to establish lineage and stage",
                    "Seek expert review for borderline morphology"
                ]
            }
        ]
    },
    "Benign": {
        "base_info": {
            "note": "No malignant blast features detected; model output suggests benign.",
            "recommendation": "If clinical suspicion persists, correlate with CBC, smear review, and further testing."
        },
        "dynamic_suggestions": [
            {
                "condition": "elevated_confidence",
                "insights": [
                    "No hallmark blast features (fine chromatin, nucleoli, high N:C) observed",
                    "Findings may represent non-malignant cells or artifacts"
                ]
            },
            {
                "condition": "moderate_confidence",
                "insights": [
                    "Some atypia present but insufficient for blast classification",
                    "Consider manual differential and clinical correlation"
                ]
            },
            {
                "condition": "low_confidence",
                "insights": [
                    "Image quality limits assessment",
                    "Repeat imaging or expert smear review advised"
                ]
            }
        ]
    }
}

# --- Cell Information rendering (patched) ---
def render_cell_information(predicted_cell_type):
    cell_info = DYNAMIC_CELL_INFO[predicted_cell_type]["base_info"]
    if "stage" in cell_info:  # malignant
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.markdown(f"""
            <div class="prediction-box">
                <h4>📊 {predicted_cell_type} Overview</h4>
                <p><strong>Stage:</strong> {cell_info['stage']}</p>
                <p><strong>Presence:</strong> {cell_info['presence']}</p>
                <p><strong>Typical Immunophenotype:</strong></p>
                <ul>
                    {"".join([f"<li>{m}</li>" for m in cell_info['typical_immunophenotype']])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with info_col2:
            st.markdown(f"""
            <div class="prediction-box">
                <h4>🔍 Morphology (expected)</h4>
                <ul>
                    {"".join([f"<li>{f}</li>" for f in cell_info['morphology_features']])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
    else:  # benign
        st.markdown(f"""
        <div class="prediction-box">
            <h4>📚 Benign Overview</h4>
            <p>{cell_info['note']}</p>
            <p><em>{cell_info['recommendation']}</em></p>
        </div>
        """, unsafe_allow_html=True)

# (rest of your original app.py code continues here; call render_cell_information() where needed)

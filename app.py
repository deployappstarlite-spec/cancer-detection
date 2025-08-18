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

# Dark Abstract CSS Theme
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 25%, #16213e 50%, #0f0f23 75%, #000000 100%);
        color: #ffffff;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
    }
    
    .analysis-card {
        background: rgba(20, 20, 40, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .analysis-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #f5576c);
        background-size: 200% 100%;
        animation: gradient-shift 3s ease infinite;
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .prediction-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 1px solid rgba(102, 126, 234, 0.4);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(15px);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.1);
    }
    
    .suggestion-box {
        background: linear-gradient(135deg, rgba(240, 147, 251, 0.1) 0%, rgba(245, 87, 108, 0.1) 100%);
        border: 1px solid rgba(240, 147, 251, 0.4);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(15px);
        box-shadow: 0 8px 32px rgba(240, 147, 251, 0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 152, 0, 0.1) 100%);
        border: 1px solid rgba(255, 193, 7, 0.4);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(15px);
        box-shadow: 0 8px 32px rgba(255, 193, 7, 0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(40, 167, 69, 0.1) 0%, rgba(25, 135, 84, 0.1) 100%);
        border: 1px solid rgba(40, 167, 69, 0.4);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(15px);
        box-shadow: 0 8px 32px rgba(40, 167, 69, 0.1);
    }
    
    .error-box {
        background: linear-gradient(135deg, rgba(220, 53, 69, 0.1) 0%, rgba(214, 51, 132, 0.1) 100%);
        border: 1px solid rgba(220, 53, 69, 0.4);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(15px);
        box-shadow: 0 8px 32px rgba(220, 53, 69, 0.1);
    }
    
    .metric-card {
        background: rgba(30, 30, 60, 0.6);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.2);
    }
    
    .glow-text {
        text-shadow: 0 0 10px rgba(102, 126, 234, 0.8);
        color: #667eea;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .sidebar .stSelectbox > div > div {
        background: rgba(30, 30, 60, 0.8);
        border: 1px solid rgba(102, 126, 234, 0.3);
        color: white;
    }
    
    .floating-particle {
        position: fixed;
        pointer-events: none;
        z-index: -1;
        opacity: 0.1;
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
</style>
""", unsafe_allow_html=True)

# Enhanced cell classification
BLOOD_CELL_CLASSES = {
    0: "Malignant (early pre-B) lymphoblast",
    1: "Malignant (pre-B) lymphoblast",
    2: "Malignant (pro-B) lymphoblast",
    3: "Benign"
}

# Dynamic medical information
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
        "dynamic_suggestions": []
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
        "dynamic_suggestions": []
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
        "dynamic_suggestions": []
    },
    "Benign": {
        "base_info": {
            "note": "No malignant blast features detected; model output suggests benign.",
            "recommendation": "If clinical suspicion persists, correlate with CBC, smear review, and further testing."
        },
        "dynamic_suggestions": []
    }
}

def calculate_image_hash(image_bytes):
    """Calculate unique hash for image to ensure different responses"""
    return hashlib.md5(image_bytes).hexdigest()

def analyze_image_characteristics(image):
    """Analyze image characteristics for dynamic suggestions"""
    img_array = np.array(image)
    
    characteristics = {
        "brightness": np.mean(img_array),
        "contrast": np.std(img_array),
        "sharpness": cv2.Laplacian(img_array, cv2.CV_64F).var() if len(img_array.shape) == 2 else cv2.Laplacian(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var(),
        "dominant_color": np.mean(img_array, axis=(0,1)) if len(img_array.shape) == 3 else np.mean(img_array),
        "texture_complexity": np.std(img_array)
    }
    
    return characteristics

def get_dynamic_suggestions(cell_type, confidence, image_hash, image_characteristics):
    """Generate dynamic suggestions based on cell type, confidence, and image characteristics"""
    if cell_type not in DYNAMIC_CELL_INFO:
        return ["Standard analysis completed", "Consider expert review for unusual findings"]
    
    cell_info = DYNAMIC_CELL_INFO[cell_type]
    
    # Determine confidence category
    if confidence > 0.8:
        condition = "elevated_confidence"
    elif confidence > 0.5:
        condition = "moderate_confidence"
    else:
        condition = "low_confidence"
    
    # Get base suggestions for the condition
    base_suggestions = []
    for suggestion_set in cell_info["dynamic_suggestions"]:
        if suggestion_set["condition"] == condition:
            base_suggestions = suggestion_set["insights"]
            break
    
    # Add image-specific insights
    image_insights = []
    
    # Brightness-based insights
    if image_characteristics["brightness"] < 80:
        image_insights.append("Image appears underexposed - consider increasing illumination")
    elif image_characteristics["brightness"] > 180:
        image_insights.append("Image may be overexposed - adjust lighting for better contrast")
    
    # Contrast-based insights
    if image_characteristics["contrast"] < 30:
        image_insights.append("Low contrast detected - enhance staining or adjust microscope settings")
    elif image_characteristics["contrast"] > 80:
        image_insights.append("High contrast image - excellent for morphological analysis")
    
    # Sharpness-based insights
    if image_characteristics["sharpness"] < 100:
        image_insights.append("Image appears slightly blurred - ensure proper focus")
    elif image_characteristics["sharpness"] > 500:
        image_insights.append("Sharp image quality - ideal for detailed morphological assessment")
    
    # Use image hash to select varied suggestions
    hash_int = int(image_hash[:8], 16)
    selected_suggestions = []
    
    # Select 2-3 base suggestions based on hash
    for i in range(min(3, len(base_suggestions))):
        idx = (hash_int + i) % len(base_suggestions)
        selected_suggestions.append(base_suggestions[idx])
    
    # Add 1-2 image-specific insights
    for i in range(min(2, len(image_insights))):
        idx = (hash_int + i + 3) % len(image_insights)
        selected_suggestions.append(image_insights[idx])
    
    return selected_suggestions

@st.cache_resource
def load_blood_cell_model():
    """Load the Keras blood cell classification model"""
    try:
        st.info("🚀 Initializing Neural Network...")
        # Load Keras model
        model = tf.keras.models.load_model("model4.h5")
        return model
    except Exception as e:
        st.error(f"❌ Model initialization failed: {str(e)}")
        return None

def preprocess_image(image, target_size=(224, 224)):
    """Advanced image preprocessing for Keras model"""
    img_array = np.array(image)
    
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    elif len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    img_resized = cv2.resize(img_array, target_size, interpolation=cv2.INTER_LANCZOS4)
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0).astype(np.float32)
    
    return img_batch

def predict_with_model(processed_image, model, num_classes=4):
    """Run prediction using Keras model"""
    try:
        # Run inference
        predictions = model.predict(processed_image, verbose=0)[0]
        predicted_class = np.argmax(predictions)
        confidence = float(predictions[predicted_class])
        
        return predicted_class, confidence, predictions
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        return None, None, None

def create_dark_confidence_chart(predictions, class_names):
    """Create a dark-themed confidence visualization"""
    df = pd.DataFrame({
        'Cell Type': [class_names.get(i, f'Class {i}') for i in range(len(predictions))],
        'Confidence': predictions * 100
    })
    df = df.sort_values('Confidence', ascending=True)
    
    fig = px.bar(df, x='Confidence', y='Cell Type', orientation='h',
                 title='AI Confidence Analysis',
                 color='Confidence',
                 color_continuous_scale='plasma',
                 text='Confidence')
    
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        showlegend=False,
        height=400
    )
    
    return fig

def create_dark_pie_chart(predictions, class_names):
    """Create a dark-themed pie chart"""
    df = pd.DataFrame({
        'Cell Type': [class_names.get(i, f'Class {i}') for i in range(len(predictions))],
        'Probability': predictions
    })
    
    fig = px.pie(df, values='Probability', names='Cell Type',
                 title='Probability Distribution',
                 color_discrete_sequence=['#667eea', '#764ba2', '#f093fb', '#f5576c'])
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400
    )
    
    return fig

def main():
    # Animated header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem;">
        <h1 class="main-header">🔬 Blood Cell AI Analyzer</h1>
        <p style="font-size: 1.2rem; color: #667eea; margin-top: -1rem;">
            Advanced Neural Network • Real-time Analysis • Dynamic Insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Streamlined sidebar
    st.sidebar.markdown("## 🎛️ Analysis Settings")
    
    confidence_threshold = st.sidebar.slider(
        "🎯 Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.05
    )
    
    enable_advanced_analysis = st.sidebar.checkbox(
        "🔬 Advanced Analysis",
        value=True,
        help="Enable detailed morphological analysis"
    )
    
    # Load model
    model = load_blood_cell_model()
    if model is None:
        st.stop()
    
    # Main analysis interface
    st.markdown("## 📤 Image Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload Blood Cell Image",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload a high-quality microscopic image"
    )
    
    if uploaded_file is not None:
        # Calculate image hash for dynamic responses
        image_bytes = uploaded_file.getvalue()
        image_hash = calculate_image_hash(image_bytes)
        
        # Process image
        image = Image.open(uploaded_file)
        image_characteristics = analyze_image_characteristics(image)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📸 Input Image")
            st.image(image, caption="Blood Cell Sample", use_container_width=True)
            
            # Image quality metrics
            st.markdown("### 📊 Image Quality")
            quality_col1, quality_col2 = st.columns(2)
            
            with quality_col1:
                brightness_score = min(100, max(0, (image_characteristics["brightness"] - 50) * 2))
                st.metric("Brightness", f"{brightness_score:.0f}/100")
            
            with quality_col2:
                contrast_score = min(100, image_characteristics["contrast"] * 1.5)
                st.metric("Contrast", f"{contrast_score:.0f}/100")
        
        with col2:
            st.markdown("### 🧠 AI Analysis")
            
            # Run analysis
            with st.spinner("🔄 Processing..."):
                processed_image = preprocess_image(image)
                predicted_class, confidence, all_predictions = predict_with_model(
                    processed_image, model
                )
                if predicted_class is None:
                    st.stop()
                predicted_cell_type = BLOOD_CELL_CLASSES.get(predicted_class, "Unknown")
            
            # Display results
            if confidence >= confidence_threshold:
                st.markdown(f"""
                <div class="success-box">
                    <h3 class="glow-text">🎯 Identified: {predicted_cell_type}</h3>
                    <h2 style="color: #28a745;">Confidence: {confidence:.1%}</h2>
                    <p>✅ High confidence detection</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-box">
                    <h3>⚠️ Possible: {predicted_cell_type}</h3>
                    <h2 style="color: #ffc107;">Confidence: {confidence:.1%}</h2>
                    <p>⚠️ Below threshold - manual review recommended</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Dynamic suggestions section
        st.markdown("---")
        st.markdown("## 💡 AI-Generated Insights")
        
        suggestions = get_dynamic_suggestions(
            predicted_cell_type, confidence, image_hash, image_characteristics
        )
        
        for i, suggestion in enumerate(suggestions, 1):
            st.markdown(f"""
            <div class="suggestion-box">
                <h4>🔍 Insight #{i}</h4>
                <p>{suggestion}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualizations
        st.markdown("---")
        st.markdown("## 📊 Analysis Dashboard")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            confidence_chart = create_dark_confidence_chart(all_predictions, BLOOD_CELL_CLASSES)
            st.plotly_chart(confidence_chart, use_container_width=True)
        
        with viz_col2:
            pie_chart = create_dark_pie_chart(all_predictions, BLOOD_CELL_CLASSES)
            st.plotly_chart(pie_chart, use_container_width=True)
        
        # Advanced analysis section
        if enable_advanced_analysis:
            st.markdown("---")
            st.markdown("## 🔬 Advanced Analysis")
            
            adv_col1, adv_col2, adv_col3 = st.columns(3)
            
            with adv_col1:
                st.markdown(f"""
                <div class="analysis-card">
                    <h4>🎯 Primary Detection</h4>
                    <h3 class="glow-text">{predicted_cell_type}</h3>
                    <p>Confidence: {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with adv_col2:
                second_idx = np.argsort(all_predictions)[-2]
                second_cell = BLOOD_CELL_CLASSES.get(second_idx, "Unknown")
                second_conf = all_predictions[second_idx]
                
                st.markdown(f"""
                <div class="analysis-card">
                    <h4>🥈 Secondary Option</h4>
                    <h3 class="glow-text">{second_cell}</h3>
                    <p>Confidence: {second_conf:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with adv_col3:
                uncertainty = 1 - confidence
                st.markdown(f"""
                <div class="analysis-card">
                    <h4>📊 Uncertainty</h4>
                    <h3 class="glow-text">{uncertainty:.1%}</h3>
                    <p>Prediction reliability</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Cell information
        if predicted_cell_type in DYNAMIC_CELL_INFO:
            st.markdown("---")
            st.markdown("## 📚 Cell Information")

            cell_info = DYNAMIC_CELL_INFO[predicted_cell_type]["base_info"]

            if "stage" in cell_info:  # malignant only
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
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h4>🔍 Morphology (expected)</h4>
                        <ul>
                            {"".join([f"<li>{f}</li>" for f in cell_info['morphology_features']])}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            else:  # benign only
                st.markdown(f"""
                <div class="prediction-box">
                    <h4>📚 Benign Overview</h4>
                    <p>{cell_info['note']}</p>
                    <p><em>{cell_info['recommendation']}</em></p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="prediction-box">
                    <h4>🔍 Morphology (expected)</h4>
                    <ul>
                        {"".join([f"<li>{feature}</li>" for feature in cell_info['morphology_features']])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: rgba(20, 20, 40, 0.5); border-radius: 15px; margin-top: 2rem;">
        <p style="color: #667eea; font-size: 0.9rem;">
            ⚠️ <strong>For Research & Educational Use Only</strong><br>
            Always consult healthcare professionals for medical decisions
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
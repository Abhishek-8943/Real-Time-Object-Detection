"""
Pascal VOC2012 Object Classifier - Streamlit Web Application
Real-time image classification with pre-trained deep learning model
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import pickle
import os
import matplotlib.pyplot as plt
from io import BytesIO
import time

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="VOC Object Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        padding: 0.5rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING AND CACHING
# ============================================================================

@st.cache_resource
def load_model_and_encoder(model_path, encoder_path):
    """Load trained model and label encoder (cached)"""
    try:
        model = keras.models.load_model(model_path)
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        return model, label_encoder
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Convert to RGB if necessary
    if len(img_array.shape) == 2:  # Grayscale
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Resize
    img_resized = cv2.resize(img_array, target_size)
    
    # Normalize
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch, img_array

def predict_image(model, label_encoder, image, top_k=5):
    """Make prediction on image"""
    # Preprocess
    img_batch, original_img = preprocess_image(image)
    
    # Predict
    start_time = time.time()
    predictions = model.predict(img_batch, verbose=0)[0]
    inference_time = time.time() - start_time
    
    # Get top-k predictions
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    results = [
        {
            'class': label_encoder.classes_[idx],
            'confidence': float(predictions[idx]),
            'percentage': float(predictions[idx] * 100)
        }
        for idx in top_indices
    ]
    
    return results, original_img, inference_time

def create_prediction_chart(results):
    """Create bar chart of predictions"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classes = [r['class'] for r in results]
    confidences = [r['confidence'] for r in results]
    
    # Color bars based on confidence
    colors = ['#28a745' if c > 0.7 else '#ffc107' if c > 0.4 else '#dc3545' 
              for c in confidences]
    
    bars = ax.barh(classes, confidences, color=colors, alpha=0.8)
    ax.set_xlabel('Confidence', fontsize=12, fontweight='bold')
    ax.set_title('Top 5 Predictions', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.grid(axis='x', alpha=0.3)
    
    # Add percentage labels
    for bar, conf in zip(bars, confidences):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{conf*100:.1f}%', ha='left', va='center', 
                fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig

def get_confidence_color(confidence):
    """Get CSS class based on confidence level"""
    if confidence > 0.7:
        return "confidence-high"
    elif confidence > 0.4:
        return "confidence-medium"
    else:
        return "confidence-low"

# ============================================================================
# WEBCAM FUNCTIONS
# ============================================================================

def process_webcam_frame(frame, model, label_encoder):
    """Process single webcam frame"""
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    # Predict
    results, _, _ = predict_image(model, label_encoder, pil_image, top_k=3)
    
    # Draw predictions on frame
    y_offset = 30
    for i, result in enumerate(results):
        text = f"{i+1}. {result['class']}: {result['percentage']:.1f}%"
        color = (0, 255, 0) if result['confidence'] > 0.7 else (255, 165, 0)
        cv2.putText(frame, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += 35
    
    return frame, results

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 VOC Object Classifier</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time image classification using Deep Learning</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Model configuration
    st.sidebar.header("Model Configuration")
    model_path = st.sidebar.text_input(
        "Model Path", 
        value=r"D:\L P U\int lpu\CV_Project\Real Time Detection\voc_classifier_model.h5",
        help="Path to the trained model file"
    )
    encoder_path = st.sidebar.text_input(
        "Label Encoder Path", 
        value=r"D:\L P U\int lpu\CV_Project\Real Time Detection\label_encoder.pkl",
        help="Path to the label encoder file"
    )
    
    # Prediction settings
    st.sidebar.header("Prediction Settings")
    top_k = st.sidebar.slider(
        "Number of Top Predictions", 
        min_value=1, 
        max_value=10, 
        value=5,
        help="Show top K predictions"
    )
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold (%)", 
        min_value=0, 
        max_value=100, 
        value=10,
        help="Minimum confidence to display"
    )
    
    # Load model
    if not os.path.exists(model_path):
        st.error(f"⚠️ Model file not found: {model_path}")
        st.info("Please train the model first using the training script, or update the model path in the sidebar.")
        return
    
    if not os.path.exists(encoder_path):
        st.error(f"⚠️ Label encoder file not found: {encoder_path}")
        st.info("Please ensure the label encoder file exists, or update the path in the sidebar.")
        return
    
    with st.spinner("Loading model..."):
        model, label_encoder = load_model_and_encoder(model_path, encoder_path)
    
    if model is None or label_encoder is None:
        st.error("Failed to load model. Please check the file paths.")
        return
    
    st.sidebar.success("✅ Model loaded successfully!")
    st.sidebar.info(f"Classes: {len(label_encoder.classes_)}")
    
    # Show available classes
    with st.sidebar.expander("📋 Available Classes"):
        for i, cls in enumerate(label_encoder.classes_, 1):
            st.write(f"{i}. {cls}")
    
    # Main content - Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📷 Upload Image", 
        "📹 Webcam", 
        "📁 Batch Processing",
        "ℹ️ About"
    ])
    
    # ========================================================================
    # TAB 1: UPLOAD IMAGE
    # ========================================================================
    with tab1:
        st.header("Upload Image for Classification")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image...", 
                type=['jpg', 'jpeg', 'png'],
                help="Upload an image to classify"
            )
            
            if uploaded_file is not None:
                # Load image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Predict button
                if st.button("🔍 Classify Image", key="classify_btn"):
                    with st.spinner("Classifying..."):
                        results, original_img, inference_time = predict_image(
                            model, label_encoder, image, top_k=top_k
                        )
                        
                        # Store results in session state
                        st.session_state['results'] = results
                        st.session_state['inference_time'] = inference_time
        
        with col2:
            if 'results' in st.session_state:
                results = st.session_state['results']
                inference_time = st.session_state['inference_time']
                
                st.subheader("🎯 Prediction Results")
                
                # Top prediction
                top_pred = results[0]
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>Top Prediction</h2>
                    <h1 style="color: #1f77b4;">{top_pred['class'].upper()}</h1>
                    <h3 class="{get_confidence_color(top_pred['confidence'])}">
                        Confidence: {top_pred['percentage']:.2f}%
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.metric("⚡ Inference Time", f"{inference_time*1000:.2f} ms")
                
                # All predictions
                st.subheader(f"Top {top_k} Predictions")
                for i, result in enumerate(results, 1):
                    if result['percentage'] >= confidence_threshold:
                        conf_class = get_confidence_color(result['confidence'])
                        st.markdown(f"""
                        **{i}. {result['class']}** - 
                        <span class="{conf_class}">{result['percentage']:.2f}%</span>
                        """, unsafe_allow_html=True)
                        st.progress(result['confidence'])
                
                # Chart
                st.subheader("📊 Confidence Distribution")
                fig = create_prediction_chart(results)
                st.pyplot(fig)
    
    # ========================================================================
    # TAB 2: WEBCAM
    # ========================================================================
    with tab2:
        st.header("Real-time Webcam Classification")
        st.info("📹 This feature requires camera access. Click 'Start' to begin.")
        
        enable_webcam = st.checkbox("Enable Webcam Feed")
        
        if enable_webcam:
            st.warning("⚠️ Webcam feature requires running the app locally with camera access.")
            st.code("""
# To use webcam, run this code separately:
import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    # Process frame with model
    cv2.imshow('VOC Classifier', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
            """, language="python")
        
        st.info("💡 Tip: For real-time webcam classification, use the standalone script with OpenCV.")
    
    # ========================================================================
    # TAB 3: BATCH PROCESSING
    # ========================================================================
    with tab3:
        st.header("Batch Image Processing")
        
        uploaded_files = st.file_uploader(
            "Upload multiple images", 
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload multiple images for batch processing"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} images uploaded")
            
            if st.button("🚀 Process All Images", key="batch_btn"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results_container = st.container()
                
                for idx, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {idx+1}/{len(uploaded_files)}: {file.name}")
                    
                    # Load and predict
                    image = Image.open(file)
                    results, _, _ = predict_image(model, label_encoder, image, top_k=3)
                    
                    # Display results
                    with results_container:
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.image(image, caption=file.name, use_container_width=True)
                        with col2:
                            st.write(f"**{file.name}**")
                            for i, result in enumerate(results, 1):
                                st.write(f"{i}. {result['class']}: {result['percentage']:.1f}%")
                        st.divider()
                    
                    # Update progress
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("✅ Batch processing complete!")
    
    # ========================================================================
    # TAB 4: ABOUT
    # ========================================================================
    with tab4:
        st.header("About This Application")
        
        st.markdown("""
        ### 🎯 VOC Object Classifier
        
        This application uses a **deep learning model** trained on the **Pascal VOC2012 dataset** 
        to classify objects in images.
        
        #### 📊 Dataset Information
        - **Dataset**: Pascal VOC2012
        - **Classes**: 20 object categories
        - **Architecture**: Transfer Learning with pre-trained CNN
        
        #### 🔧 Technical Details
        - **Framework**: TensorFlow/Keras
        - **Base Models**: ResNet50, VGG16, or MobileNetV2
        - **Input Size**: 224x224 pixels
        - **Output**: Softmax probability distribution
        
        #### 🚀 Features
        - ✅ Single image classification
        - ✅ Batch processing
        - ✅ Real-time confidence scores
        - ✅ Top-K predictions
        - ✅ Interactive visualizations
        
        #### 📋 Supported Classes
        """)
        
        # Display classes in columns
        classes = label_encoder.classes_
        cols = st.columns(4)
        for i, cls in enumerate(classes):
            cols[i % 4].write(f"• {cls}")
        
        st.markdown("""
        ---
        #### 💡 Usage Tips
        1. Upload clear, well-lit images for best results
        2. Model works best with objects similar to VOC dataset
        3. Higher confidence scores indicate more certain predictions
        4. Try adjusting the confidence threshold in settings
        
        #### 🔗 Resources
        - [Pascal VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/)
        - [TensorFlow Documentation](https://www.tensorflow.org/)
        - [Streamlit Documentation](https://docs.streamlit.io/)
        """)
        
        st.info("📧 For questions or issues, please refer to the documentation.")

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from models import StudentModel
import torchvision.transforms as T

# Page config
st.set_page_config(
    page_title="Astigmatism Predictor",
    page_icon="👁️",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StudentModel().to(device)
    try:
        model.load_state_dict(torch.load("student_astigmatism_model.pth", map_location=device, weights_only=False))
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error("❌ Model file not found. Please train the model first.")
        return None, device

# Image preprocessing
def preprocess_image(image):
    """Preprocess uploaded image to match training format"""
    # Convert PIL to numpy
    img = np.array(image)
    
    # Convert to RGB if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Resize to 224x224
    img = cv2.resize(img, (224, 224))
    
    # Convert to PIL for transforms
    img_pil = Image.fromarray(img)
    
    # Apply same transforms as training
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img_pil).unsqueeze(0)  # Add batch dimension
    return img_tensor

# Main app
def main():
    st.title("👁️ Astigmatism Prediction")
    st.markdown("Upload a cropped limbus image to predict astigmatism magnitude")
    
    # Load model
    model, device = load_model()
    
    if model is None:
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a limbus image",
        type=["png", "jpg", "jpeg"],
        help="Upload a cropped slitlamp limbus image"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Prediction")
            
            # Predict button
            if st.button("🔍 Predict Astigmatism", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    try:
                        # Preprocess
                        img_tensor = preprocess_image(image).to(device)
                        
                        # Predict
                        with torch.no_grad():
                            prediction, _ = model(img_tensor)
                            astigmatism = prediction.item()
                        
                        # Display result
                        st.success("✅ Prediction Complete")
                        
                        # Show prediction with color coding
                        if astigmatism < 3.0:
                            color = "🟢"
                            severity = "Low"
                        elif astigmatism < 6.0:
                            color = "🟡"
                            severity = "Moderate"
                        else:
                            color = "🔴"
                            severity = "High"
                        
                        st.metric(
                            label="Predicted Astigmatism",
                            value=f"{astigmatism:.2f} D",
                            delta=severity,
                            delta_color="off"
                        )
                        
                        st.markdown(f"{color} **Severity:** {severity}")
                        
                        # Clinical interpretation
                        with st.expander("ℹ️ Clinical Interpretation"):
                            if astigmatism < 1.5:
                                st.info("Minimal astigmatism. Excellent graft outcome.")
                            elif astigmatism < 3.0:
                                st.info("Low astigmatism. Good graft outcome.")
                            elif astigmatism < 6.0:
                                st.warning("Moderate astigmatism. May require correction.")
                            else:
                                st.error("High astigmatism. Consider intervention.")
                    
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.caption("Model: LUPI Student Network | Trained on PK/DALK graft images")

if __name__ == "__main__":
    main()

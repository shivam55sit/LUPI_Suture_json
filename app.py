import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from models import StudentModel
import torchvision.transforms as T
from batch_limbus_crop import get_cropper

# Page config
st.set_page_config(
    page_title="Astigmatism Predictor",
    page_icon="👁️",
    layout="wide"
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


# Load cropping model
@st.cache_resource
def load_cropper():
    try:
        return get_cropper("model_limbus_crop_unetpp_weighted.pth")
    except Exception as e:
        st.error(f"❌ Error loading cropping model: {e}")
        return None

def crop_limbus(image, cropper):
    """
    Crop the limbus region using the deep learning model.
    """
    if cropper is None:
        return None, None, False
        
    return cropper.crop_image(image)

def preprocess_image(image):
    """Preprocess cropped limbus image for model input"""
    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Resize to 224x224
    img_resized = cv2.resize(image, (224, 224))
    
    # Convert to PIL for transforms
    img_pil = Image.fromarray(img_resized)
    
    # Apply same transforms as training
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img_pil).unsqueeze(0)  # Add batch dimension
    return img_tensor

# Main app
def main():
    st.title("👁️ Astigmatism Prediction from Slitlamp Images")
    st.markdown("Upload a slitlamp image to automatically detect and analyze the limbus region")
    
    # Load model
    model, device = load_model()
    
    if model is None:
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Slitlamp Image",
        type=["png", "jpg", "jpeg"],
        help="Upload a full slitlamp image - limbus will be automatically detected and cropped"
    )
    
    # Load cropper
    cropper = load_cropper()
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        
        # Create columns for display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📸 Original Image")
            st.image(image, use_container_width=True)
        
        # Crop limbus
        with st.spinner("Detecting limbus..."):
            cropped_img, vis_img, success = crop_limbus(image, cropper)
        
        with col2:
            st.subheader("🎯 Detected Limbus")
            if success:
                st.image(vis_img, use_container_width=True, caption="Green overlay: detected limbus region")
            else:
                st.warning("⚠️ Segmentation failed.")
                if vis_img is not None:
                    st.image(vis_img, use_container_width=True)
        
        with col3:
            st.subheader("✂️ Cropped Limbus")
            st.image(cropped_img, use_container_width=True)
        
        # Predict button
        st.markdown("---")
        if st.button("🔍 Predict Astigmatism", type="primary", use_container_width=True):
            with st.spinner("Analyzing cropped limbus..."):
                try:
                    # Preprocess
                    img_tensor = preprocess_image(cropped_img).to(device)
                    
                    # Predict
                    with torch.no_grad():
                        prediction, _ = model(img_tensor)
                        astigmatism = prediction.item()
                    
                    # Display result
                    st.success("✅ Prediction Complete")
                    
                    # Create result columns
                    res_col1, res_col2, res_col3 = st.columns(3)
                    
                    with res_col1:
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
                            value=f"{astigmatism:.2f} D"
                        )
                    
                    with res_col2:
                        st.metric(
                            label="Severity Level",
                            value=severity
                        )
                        st.markdown(f"### {color}")
                    
                    with res_col3:
                        # Clinical interpretation
                        if astigmatism < 1.5:
                            st.info("**Minimal astigmatism**\n\nExcellent graft outcome.")
                        elif astigmatism < 3.0:
                            st.info("**Low astigmatism**\n\nGood graft outcome.")
                        elif astigmatism < 6.0:
                            st.warning("**Moderate astigmatism**\n\nMay require correction.")
                        else:
                            st.error("**High astigmatism**\n\nConsider intervention.")
                
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.caption("🔬 Model: LUPI Student Network | 📊 Trained on PK/DALK graft images")

if __name__ == "__main__":
    main()

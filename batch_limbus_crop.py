import os
import cv2
import torch
import numpy as np
from inference_utils import load_model, predict_masks, get_cropped_roi

class LimbusCropper:
    def __init__(self, model_path="model_limbus_crop_unetpp_weighted.pth", device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model_path = model_path
        self.model = None
        self.img_size = None
        self.idx_limbus = None
        
        # Load model immediately
        self._load_model()
    
    def _load_model(self):
        print(f"Loading cropping model from {self.model_path} on {self.device}...")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        try:
            self.model, _, self.idx_limbus, self.img_size = load_model(self.model_path, self.device)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def crop_image(self, image):
        """
        Crop limbus from a single image (numpy array or PIL Image)
        Args:
            image: numpy array (BGR or RGB) or PIL Image
        Returns:
            cropped_roi: Cropped limbus image (numpy array)
            mask_vis: Visualization of the mask (optional)
            success: Boolean indicating if crop was successful
        """
        # Convert PIL to numpy if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
            # Convert RGB to BGR for OpenCV compatibility if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Ensure image is BGR for model
        bgr = image.copy()
        
        try:
            with torch.no_grad():
                masks = predict_masks(self.model, bgr, self.img_size, self.device)
                crop_mask = masks[self.idx_limbus]
                
                # Extract ROI
                roi = get_cropped_roi(bgr, crop_mask)
                
                if roi is not None:
                    # ROI is in BGR, convert to RGB for display/processing
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    
                    # Create visualization overlay
                    mask_vis = bgr.copy()
                    # Create green overlay for mask
                    colored_mask = np.zeros_like(bgr)
                    colored_mask[crop_mask > 0.5] = [0, 255, 0]  # Green channel
                    mask_vis = cv2.addWeighted(mask_vis, 0.7, colored_mask, 0.3, 0)
                    mask_vis_rgb = cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB)
                    
                    return roi_rgb, mask_vis_rgb, True
                else:
                    return None, None, False
                    
        except Exception as e:
            print(f"Error cropping image: {e}")
            return None, None, False

# Function for simpler usage
def get_cropper(model_path="model_limbus_crop_unetpp_weighted.pth"):
    return LimbusCropper(model_path)


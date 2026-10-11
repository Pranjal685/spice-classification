"""
src/explain.py
--------------
Explainable AI (XAI) Module using Grad-CAM.
Generates an attention heatmap overlaid on the input image using PyTorch hooks.
"""

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from pathlib import Path
from src.model import build_model
import torchvision.transforms as transforms

# Global vars for hooks
activations = None
gradients = None

def _forward_hook(module, input, output):
    global activations
    activations = output

def _backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

def init_cam_model(model_path: Path, device: torch.device, num_classes: int = 14) -> torch.nn.Module:
    """Loads PyTorch model and registers Grad-CAM hooks to the final conv layer."""
    model = build_model("mobilenet_v3_small", num_classes=num_classes, pretrained=False)
    state_dict = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # For MobileNetV3-Small, features[-1] is the final block
    target_layer = model.features[-1]
    
    target_layer.register_forward_hook(_forward_hook)
    target_layer.register_full_backward_hook(_backward_hook)
    return model

def generate_gradcam(
    model: torch.nn.Module, 
    image: Image.Image, 
    device: torch.device,
    target_class_idx: int = None
) -> tuple[np.ndarray, int, float]:
    """
    Generates a Grad-CAM overlay heatmap.
    Returns: (blended_image_bgr_array, predicted_class_idx, confidence)
    """
    global activations, gradients
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    input_tensor.requires_grad = True

    model.zero_grad()
    
    # Forward pass
    logits = model(input_tensor)
    probs = F.softmax(logits, dim=1)
    
    if target_class_idx is None:
        target_class_idx = logits.argmax(dim=1).item()
        
    confidence = probs[0, target_class_idx].item()
    
    # Backward pass for the target class
    class_score = logits[0, target_class_idx]
    class_score.backward()
    
    # Compute Grad-CAM
    # gradients: (1, C, H, W), activations: (1, C, H, W)
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3]) # (C,)
    
    # Use clone to avoid modifying original activations graph
    act_clone = activations.clone()
    for i in range(act_clone.size(1)):
        act_clone[:, i, :, :] *= pooled_gradients[i]
        
    heatmap = torch.mean(act_clone, dim=1).squeeze().cpu().detach().numpy()
    
    # ReLU
    heatmap = np.maximum(heatmap, 0)
    # Normalize
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
        
    # Resize and Overlay
    heatmap_resized = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    original_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    
    # original_bgr and colormap have same dimensions now
    blended = cv2.addWeighted(original_bgr, 0.6, colormap, 0.4, 0)
    
    # Clear memory to prevent graph leaks
    del input_tensor
    del logits
    del class_score
    activations = None
    gradients = None
    
    return blended, target_class_idx, confidence, colormap

def encode_image_base64(img_bgr: np.ndarray) -> str:
    """Encodes BGR numpy array to Base64 JPEG."""
    _, buffer = cv2.imencode('.jpg', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torch import nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
import os

# Set page configuration
st.set_page_config(page_title="Mushroom Classifier", layout="wide")

# Custom center crop class from notebook
class CenterCrop(torch.nn.Module):
    def __init__(self, size=None, ratio="1:1"):
        super().__init__()
        self.size = size
        self.ratio = ratio

    def forward(self, img):
        if self.size is None:
            if isinstance(img, torch.Tensor):
                h, w = img.shape[-2:]
            else:
                w, h = img.size
            ratio = self.ratio.split(":")
            ratio = float(ratio[0]) / float(ratio[1])
            ratioed_w = int(h * ratio)
            ratioed_h = int(w / ratio)
            if w>=h:
                if ratioed_h <= h:
                    size = (ratioed_h, w)
                else:
                    size = (h, ratioed_w)
            else:
                if ratioed_w <= w:
                    size = (h, ratioed_w)
                else:
                    size = (ratioed_h, w)
        else:
            size = self.size
        return transforms.functional.center_crop(img, size)

# Class names from the notebook
class_names = [
    'Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma', 
    'Hygrocybe', 'Lactarius', 'Russula', 'Suillus'
]

def load_mushroom_model(model_path, num_classes=len(class_names)):
    """
    Load a saved mushroom classification model
    
    Args:
        model_path: Path to the saved model file
        num_classes: Number of mushroom classes
    
    Returns:
        Loaded model ready for inference
    """
    # Set device for model loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"Using device: {device}")
    
    # Create a new model with the same architecture
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load the saved parameters
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, device

def predict_mushroom(model, image, class_names, device, top_k=3):
    """
    Predict mushroom class from an image
    
    Args:
        model: Loaded model
        image: PIL Image to classify
        class_names: List of class names
        device: Device to run inference on (cuda/cpu)
        top_k: Number of top predictions to return
    
    Returns:
        Dictionary with prediction results
    """
    # Apply the same preprocessing as during testing
    transform = transforms.Compose([
        CenterCrop(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Transform and add batch dimension
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)
    
    # Prepare results
    predictions = []
    for i in range(top_k):
        class_idx = top_indices[0][i].item()
        prob = top_probs[0][i].item()
        predictions.append({
            'class': class_names[class_idx],
            'probability': prob,
            'percentage': f"{prob * 100:.2f}%"
        })
    
    return {
        'predictions': predictions,
        'top_class': class_names[top_indices[0][0].item()],
        'top_probability': top_probs[0][0].item()
    }

def main():
    # App title and description
    st.title("🍄 Mushroom Classification App")
    st.markdown("""
    This application identifies mushroom species from images using a deep learning model.
    Upload a mushroom image to get a classification result.
    """)
    
    # Sidebar with information
    st.sidebar.header("About")
    st.sidebar.info("""
    This app uses a ResNet50 model trained on mushroom images.
    It can recognize the following mushroom genera:
    - Agaricus
    - Amanita
    - Boletus
    - Cortinarius
    - Entoloma
    - Hygrocybe
    - Lactarius
    - Russula
    - Suillus
    """)
    
    # Model path input (with default value)
    default_model_path = "mushroom_optimized_model.pt"
    model_path = st.sidebar.text_input(
        "Model Path", 
        value=default_model_path,
        help="Path to the trained model file"
    )
    
    # Load model button
    load_model_button = st.sidebar.button("Load Model")
    model_loaded = False
    
    # Session state to store the model
    if 'model' not in st.session_state:
        st.session_state.model = None
        st.session_state.device = None
    
    # Load model if button is pressed
    if load_model_button:
        try:
            with st.spinner("Loading model..."):
                st.session_state.model, st.session_state.device = load_mushroom_model(model_path)
            st.sidebar.success("Model loaded successfully!")
            model_loaded = True
        except FileNotFoundError:
            st.sidebar.error(f"Model not found at {model_path}")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {str(e)}")
    
    # If model was previously loaded, show success message
    elif st.session_state.model is not None:
        st.sidebar.success("Model loaded successfully!")
        model_loaded = True
    
    # File uploader
    st.subheader("Upload a Mushroom Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Layout columns
    col1, col2 = st.columns([1, 1])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Make prediction if model is loaded
        if model_loaded:
            with st.spinner("Classifying..."):
                try:
                    results = predict_mushroom(
                        st.session_state.model, 
                        image, 
                        class_names, 
                        st.session_state.device
                    )
                    
                    # Display results
                    with col2:
                        st.subheader("Prediction Results")
                        
                        # Display top prediction with styling
                        st.markdown(f"""
                        <div style="padding: 10px; border-radius: 5px; background-color: dark; margin-bottom: 5px;">
                            <h3 style="color: #1f77b4; margin: 0;">
                                {results['top_class']}
                            </h3>
                            <p style="font-size: 1.2em; margin: 0;">
                                Confidence: {results['predictions'][0]['percentage']}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Create bar chart for top predictions
                        fig, ax = plt.subplots(figsize=(10, 5))
                        
                        classes = [pred['class'] for pred in results['predictions']]
                        probs = [pred['probability'] for pred in results['predictions']]
                        
                        # Horizontal bar chart
                        bars = ax.barh(classes, probs, color='skyblue')
                        ax.set_xlim(0, 1.0)
                        ax.set_xlabel('Probability')
                        ax.set_title('Top Predictions')
                        
                        # Add percentage labels
                        for bar in bars:
                            width = bar.get_width()
                            label_x_pos = width + 0.01
                            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1%}',
                                   va='center')
                        
                        st.pyplot(fig)
                        
                        # Display data table
                        st.subheader("Detailed Results")
                        result_df = {
                            "Class": [pred['class'] for pred in results['predictions']],
                            "Probability": [f"{pred['probability']:.4f}" for pred in results['predictions']],
                            "Percentage": [pred['percentage'] for pred in results['predictions']]
                        }
                        st.table(result_df)
                
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
        else:
            with col2:
                st.warning("Please load the model first using the sidebar.")
    
    # Instructions when no file is uploaded
    else:
        with st.container():
            st.info("Please upload an image to get started.")
    
if __name__ == "__main__":
    main()

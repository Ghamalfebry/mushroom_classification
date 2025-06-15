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
    print(f"Using device: {device}")
    
    # Create a new model with the same architecture
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load the saved parameters
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, device

def predict_mushroom(model, image_path, class_names, device, top_k=3):
    """
    Predict mushroom class from an image
    
    Args:
        model: Loaded model
        image_path: Path to the mushroom image
        class_names: List of class names
        device: Device to run inference on (cuda/cpu)
        top_k: Number of top predictions to return
    
    Returns:
        Dictionary with prediction results
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    
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

# Example usage
if __name__ == "__main__":
    # Path to saved model
    model_path = "/content/drive/MyDrive/Skripsi-ghamal/mushroom_optimized_model.pt"  # update with your actual path if different
    
    try:
        # Load the model
        loaded_model, inference_device = load_mushroom_model(model_path)
        print("Model loaded successfully!")
        
        # Define a function to display prediction results with an image
        def display_prediction(image_path):
            # Make prediction
            results = predict_mushroom(loaded_model, image_path, class_names, inference_device)
            
            # Display image with predictions
            img = Image.open(image_path).convert('RGB')
            plt.figure(figsize=(10, 6))
            plt.imshow(img)
            plt.title(f"Top prediction: {results['top_class']} ({results['predictions'][0]['percentage']})")
            plt.axis('off')
            
            # Display prediction table
            print("\nPrediction Results:")
            print("-" * 50)
            print(f"{'Class':<25} | {'Probability':<10} | {'Percentage':<10}")
            print("-" * 50)
            for pred in results['predictions']:
                print(f"{pred['class']:<25} | {pred['probability']:.6f} | {pred['percentage']:<10}")
            
            return results
        
        print("\nModel is ready for predictions!")
        print("Use the display_prediction() function with your image path to get predictions.")
        
    except FileNotFoundError:
        print(f"Model file not found at {model_path}")
        print("Please provide the correct path to the saved model file.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

example_image = "/content/drive/MyDrive/Skripsi-ghamal/Mushrooms/Boletus/0001_yB5GiXfgyRU.jpg"
display_prediction(example_image)
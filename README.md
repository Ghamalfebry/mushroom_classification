# Mushroom Classification Project 🍄

This project implements advanced deep learning techniques to classify mushroom genera using computer vision. The system is capable of identifying 9 different mushroom genera with high accuracy.

## Features

- Deep learning-based mushroom classification using ResNet50 architecture
- Hyperparameter tuning using Particle Swarm Optimization (PSO)
- Interactive Streamlit web interface for real-time classification
- Comprehensive EDA and model performance analysis
- Responsive UI with visualization of prediction confidence

## Dataset

The model is trained on a dataset containing images from 9 mushroom genera:

- Agaricus
- Amanita
- Boletus
- Cortinarius
- Entoloma
- Hygrocybe
- Lactarius
- Russula
- Suillus

## Project Structure

```
.
├── app.py                # Streamlit web app
├── model.py              # Model definition and training code
├── requirements.txt       # Python package dependencies
├── README.md              # Project documentation
└── data                   # Dataset directory
    ├── raw                # Raw images
    └── processed          # Processed images for training
```

## Installation

1. Clone this repository
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your trained model file (`.pt`) in the project directory
2. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

3. Open your browser and go to the URL displayed in the terminal (typically http://localhost:8501)
4. Use the sidebar to load the model
5. Upload a mushroom image to get classification results

## Model Structure

The app uses a ResNet50 model trained to classify 9 different mushroom genera:

- Agaricus
- Amanita
- Boletus
- Cortinarius
- Entoloma
- Hygrocybe
- Lactarius
- Russula
- Suillus

## Requirements

See `requirements.txt` for the full list of dependencies.

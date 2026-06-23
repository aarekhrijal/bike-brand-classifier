# Bike Brand Classifier

An image classifier that identifies motorcycle brands (Honda, Yamaha, KTM) using deep learning.

## Approach
- Collected ~90-100 images per brand using image search
- Trained using fastai with ResNet34 (transfer learning)
- Fine-tuned for 5 epochs

## Results
- Final error rate: 1.8%
- Validation accuracy: ~98%

## Tech Stack
- Python, fastai, PyTorch
- Trained on Kaggle (free GPU)

## Status
Model trained successfully. Deployment to Hugging Face Spaces in progress (debugging Python/library version compatibility issues between training and deployment environments).

## Files
- `notebook.py` - training code
- `app.py` - Gradio deployment interface
- `bike_classifier.pkl` - trained model

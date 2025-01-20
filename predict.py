import argparse
import json
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms

def arg_parse():
    parser = argparse.ArgumentParser(description='Image Classifier Prediction')

    parser.add_argument('image_dir', help='Path to the input image')
    parser.add_argument('checkpoint_dir', help='Path to the checkpoint file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    parser.add_argument('--top_k', type=int, help='Return top K most likely classes', default=5)
    parser.add_argument('--category_names', help='Path to the category names JSON file', default='cat_to_name.json')

    return parser.parse_args()

def load_model(arch):
    # Use getattr to load the model architecture
    if hasattr(models, arch):
        model = getattr(models, arch)(pretrained=True)
    else:
        raise ValueError(f"Model architecture '{arch}' is not supported.")
    
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    return model

def initialize_classifier(model, hidden_units=150, out_features=102):
    # Initialize the classifier based on the model architecture
    if hasattr(model, 'classifier'):
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, out_features),
            nn.LogSoftmax(dim=1)
        )
    elif hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, out_features),
            nn.LogSoftmax(dim=1)
        )
    else:
        raise AttributeError("Model does not have 'classifier' or 'fc' attribute.")
    
    return model

def load_checkpoint(checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
    output_features = checkpoint['output_features']

    # Load the model architecture
    model = load_model(arch)
    model = initialize_classifier(model, hidden_units, output_features)

    # Load the model state dict
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def process_image(image_path):
    # Process the input image
    image_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    pil_image = Image.open(image_path).convert('RGB')
    tensor_image = image_transforms(pil_image)
    return tensor_image

def predict(image_path, model, topk, device):
    # Predict the top K classes
    model.to(device)
    model.eval()

    input_img = process_image(image_path)
    input_img = input_img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_img)
        probabilities = F.softmax(output, dim=1)
        top_probs, top_indices = probabilities.topk(topk)

    # Convert tensors to numpy arrays
    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]

    # Map indices to class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices]

    return top_probs, top_classes

def main():
    args = arg_parse()

    # Set device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    # Load category names
    with open(args.category_names, 'r') as json_file:
        cat_to_name = json.load(json_file)

    # Load the model from checkpoint
    model = load_checkpoint(args.checkpoint_dir)

    # Perform prediction
    probabilities, classes = predict(args.image_dir, model, args.top_k, device)

    # Map class indices to names
    class_names = [cat_to_name[cls] for cls in classes]

    # Print results
    for i in range(args.top_k):
        print(f"{class_names[i]} with a probability of {probabilities[i]:.4f}")

if __name__ == '__main__':
    main()
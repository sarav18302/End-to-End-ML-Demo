import cv2
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms,models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import shutil
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Evaluate the model
def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds)

    return accuracy, conf_matrix, class_report

# Load best model and evaluate
model.load_state_dict(torch.load('best_resnet_binary_model.pth'))
accuracy, conf_matrix, class_report = evaluate_model(model, val_loader)

print(f"Validation Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)



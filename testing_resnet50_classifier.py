import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import resnet50
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import time

def resnet50_model(num_classes):
    # Load pre-trained ResNet50 model
    model = resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes) 
    model = model.to(device)
    print(model)
    return model

def test_model(test_dataset, num_classes, device, trained_model_path):
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    # Load the trained model
    model = resnet50_model(num_classes)
    model.load_state_dict(torch.load(trained_model_path))
    model.eval()

    # Initialize variables for evaluation
    true_labels = []
    predicted_labels = []
    inference_times = []

    # Predict outputs for the test dataset
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()
            
            inference_times.append(end_time - start_time)
            true_labels.extend(labels.numpy())
            predicted_labels.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    # Calculate overall accuracy
    overall_accuracy = np.mean(np.array(true_labels) == np.array(predicted_labels))

    # Calculate average inference time
    average_inference_time = np.mean(inference_times)

    # Calculate precision, recall, and F1 score
    precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='macro')

    # Plot confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig("./outputs/test_classifier_confusion_matrix.jpg")

    # Calculate class-wise accuracy
    class_accuracy = np.diag(cm) / np.sum(cm, axis=1)

    # Plot class-wise accuracy
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(class_accuracy)), class_accuracy)
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Class-wise Accuracy')
    plt.savefig("./outputs/test_classifier_classwise_accuracy.jpg")

    # Print evaluation metrics
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Average Inference Time: {average_inference_time:.4f} seconds")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")


if __name__ == "__main__":
    # Load the test dataset
    test_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = ImageFolder(root='./prepared_dataset/test', transform=test_transform)
    num_classes = 105
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Trained model path
    trained_model_path = "./outputs/weights/resnet50_classifier.pt"
    # Call test_model function
    test_model(test_dataset, num_classes, device, trained_model_path)

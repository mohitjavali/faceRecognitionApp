import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
import time
import os
import csv
import matplotlib.pyplot as plt

def resnet50_model(num_classes):
    # Load pre-trained ResNet50 model
    model = resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes) 
    model = model.to(device)
    print(model)
    return model

def train_model(train_dataset, test_dataset, batch_size, device, learning_rate, patience, model, pretrained_weights_path, new_weights_path):
    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=patience, verbose=True)

    # Check if previously trained weights exist
    if pretrained_weights_path is not None and os.path.exists(pretrained_weights_path):
        print("Loading previously trained weights...")
        model.load_state_dict(torch.load(pretrained_weights_path))

    # Early stopping parameters
    early_stopping = False
    best_val_loss = float('inf')
    counter = 0

    # Training loop
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    epoch_times = []

    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = correct_train / total_train
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(test_loader.dataset)
        val_acc = correct_val / total_val
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), new_weights_path)
        else:
            counter += 1
            if counter >= patience:
                early_stopping = True
                break
        
        end_time = time.time()
        epoch_time = end_time - start_time
        epoch_times.append(epoch_time)
        
        # Print epoch details
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Epoch Time: {epoch_time:.2f} s')
        
        # Save training details to CSV
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        with open('./outputs/classifier_training_details.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc', 'Epoch Time'])
            for i in range(epoch+1):
                writer.writerow([i+1, train_losses[i], val_losses[i], train_accs[i], val_accs[i], epoch_times[i]])

    # Plot train and val loss vs epochs and train and val accuracy vs epochs
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accs) + 1), train_accs, label='Train Acc')
    plt.plot(range(1, len(val_accs) + 1), val_accs, label='Val Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./outputs/classifier_training_plots.jpg')
    plt.show()

    print('Training finished.')

    if early_stopping:
        print('Early stopping triggered.')


if __name__ == "__main__":
    # Define constants
    num_classes = 105
    num_epochs = 50
    batch_size = 32
    learning_rate = 0.001
    patience = 5

    # Define transformations for the dataset
    train_transform = transforms.Compose([
        transforms.Resize((112, 112)),  # Resize to 112x112
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((112, 112)),  # Resize to 112x112
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the dataset
    train_dataset = ImageFolder(root='./prepared_dataset/train', transform=train_transform)
    test_dataset = ImageFolder(root='./prepared_dataset/test', transform=test_transform)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model
    model = resnet50_model(num_classes)

    # Grab weights from previously trained .pt file
    pretrained_weights_path = None

    os.makedirs(os.path.dirname("./outputs/weights"), exist_ok=True)
    new_weights_path = "./outputs/weights/resnet50_classifier.pt"
    train_model(train_dataset, test_dataset, batch_size, device, learning_rate, patience, model, pretrained_weights_path, new_weights_path)
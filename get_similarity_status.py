import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
import pandas as pd
from itertools import combinations

def resnet50_model_without_fc(num_classes):
    # Load pre-trained ResNet50 model
    model = resnet50(pretrained=False)
    # Remove the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Identity()
    model.fc_out_features = num_ftrs  # Store the number of features before the fully connected layer
    return model

# Function to extract features from images using the trained ResNet50 model
def extract_features(model, dataloader):
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            outputs = model(images)
            features.extend(outputs.cpu().numpy())
            labels.extend(lbls.cpu().numpy())
    return features, labels

# Load the test dataset
test_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 105

# Load the test dataset
test_dataset = ImageFolder(root='./prepared_dataset/test', transform=test_transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

# Initialize the ResNet50 model without the final fully connected layer
model = resnet50_model_without_fc(num_classes)
# Load the state_dict into the modified model (without the final fully connected layer)
state_dict = torch.load("./outputs/weights/resnet50_classifier.pt", map_location=torch.device('cpu'))
# Remove keys corresponding to the final fully connected layer
modified_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
# Load the modified state_dict into the model
model.load_state_dict(modified_state_dict, strict=False)  # Use strict=False to ignore missing keys
# Set the model to evaluation mode
model.eval()
print(model)

# Extract features from the test dataset
test_features, test_labels = extract_features(model, test_loader)

# Save image paths, feature embeddings, and labels in a CSV file
test_dataset_features = pd.DataFrame({'Image_Path': test_dataset.imgs,
                                      'Feature_Embeddings': test_features,
                                      'Labels': test_labels})
test_dataset_features.to_csv('outputs/test_dataset_features.csv', index=False)

# Generate combinations of pairs of images in the test dataset
image_combinations = list(combinations(range(len(test_dataset)), 2))

# Calculate similarity status for each pair
actual_similarity_combinations = []
for img1_idx, img2_idx in image_combinations:
    img1_label = test_dataset_features.iloc[img1_idx]['Labels']
    img2_label = test_dataset_features.iloc[img2_idx]['Labels']
    similarity_status = int(img1_label == img2_label)
    img1_path = test_dataset_features.iloc[img1_idx]['Image_Path']
    img2_path = test_dataset_features.iloc[img2_idx]['Image_Path']
    img1_embedding = test_dataset_features.iloc[img1_idx]['Feature_Embeddings']
    img2_embedding = test_dataset_features.iloc[img2_idx]['Feature_Embeddings']
    actual_similarity_combinations.append((img1_path, img1_embedding, img2_path, img2_embedding, similarity_status))

# Save actual similarity combinations to a CSV file
actual_similarity_df = pd.DataFrame(actual_similarity_combinations, columns=['Image1_Path', 'Image1_Embeddings', 'Image2_Path', 'Image2_Embeddings', 'Similarity_Status'])
actual_similarity_df.to_csv('outputs/actual_similarity_combinations.csv', index=False)

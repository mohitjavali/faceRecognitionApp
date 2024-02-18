import os
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from deepface import DeepFace
import cv2 

# Function to detect faces using FaceNet
def detect_faces_facenet_v1(input_folder, output_folder):
    mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            
            # Detect faces
            boxes, _ = mtcnn.detect(img)
            
            if boxes is not None:
                for i, box in enumerate(boxes):
                    face = img.crop(box)
                    face = face.resize((112, 112))  # Use default resampling method (antialiased)
                    output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_{i}.jpg")
                    face.save(output_path)

def detect_faces_facenet_v2(input_folder, output_folder):
    mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            
            # Detect faces
            boxes, _ = mtcnn.detect(img)
            
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Calculate cropping coordinates
                    crop_x1, crop_y1, crop_x2, crop_y2 = box.tolist()
                    
                    # Determine the size of the square region around the center
                    max_dim = max(crop_x2 - crop_x1, crop_y2 - crop_y1)
                    half_dim = max_dim // 2
                    
                    # Calculate the center of the face bounding box
                    center_x = crop_x1 + (crop_x2 - crop_x1) // 2
                    center_y = crop_y1 + (crop_y2 - crop_y1) // 2
                    
                    # Calculate the cropping coordinates to ensure the face is centered
                    crop_x1 = max(center_x - half_dim, 0)
                    crop_y1 = max(center_y - half_dim, 0)
                    crop_x2 = min(center_x + half_dim, img.width)
                    crop_y2 = min(center_y + half_dim, img.height)
                    
                    # Calculate padding
                    pad_left = max(0 - crop_x1, 0)
                    pad_right = max(crop_x2 - img.width, 0)
                    pad_top = max(0 - crop_y1, 0)
                    pad_bottom = max(crop_y2 - img.height, 0)
                    
                    # Pad the image with black borders
                    padded_img = Image.new('RGB', (img.width + pad_left + pad_right, img.height + pad_top + pad_bottom), (0, 0, 0))
                    padded_img.paste(img, (pad_left, pad_top))
                    
                    # Update cropping coordinates with padding
                    crop_x1 += pad_left
                    crop_y1 += pad_top
                    crop_x2 += pad_left
                    crop_y2 += pad_top
                    
                    # Crop the face
                    face = padded_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    
                    # Resize the face to 112x112
                    face = face.resize((112, 112))
                    
                    # Save the cropped and resized face
                    output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_{i}.jpg")
                    face.save(output_path)


# Function to detect faces using DeepFace
def detect_faces_deepface_v1(input_folder, output_folder):
    detector_name = "mtcnn"  # You can change this to other detectors supported by DeepFace if needed
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            
            # Detect faces
            obj = DeepFace.extract_faces(img, detector_backend=detector_name, enforce_detection=False)
            
            if obj is not None:
                print(f"There are {len(obj)} faces in {filename}")
                for i, face_data in enumerate(obj):
                    # Extract face area
                    facial_area = face_data["facial_area"]
                    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                    
                    # Calculate center of the face bounding box
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Determine the size of the square region around the center
                    max_dim = max(w, h)
                    half_dim = max_dim // 2
                    
                    # Calculate the cropping coordinates to ensure the face is centered
                    crop_x1 = max(center_x - half_dim, 0)
                    crop_y1 = max(center_y - half_dim, 0)
                    crop_x2 = min(center_x + half_dim, img.shape[1])
                    crop_y2 = min(center_y + half_dim, img.shape[0])
                    
                    # Crop the square region around the center
                    square_face = img[crop_y1:crop_y2, crop_x1:crop_x2]
                    
                    # Resize the face to 112x112
                    resized_face = cv2.resize(square_face, (112, 112), interpolation=cv2.INTER_AREA)
                    
                    # Save cropped and resized face
                    output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_{i}.jpg")
                    cv2.imwrite(output_path, resized_face)
                    print(f"Cropped and resized face saved to {output_path}")
            else:
                print(f"No face detected in {filename}")
                    
def detect_faces_deepface_v2(input_folder, output_folder):
    detector_name = "mtcnn"  # You can change this to other detectors supported by DeepFace if needed
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            
            # Detect faces
            obj = DeepFace.extract_faces(img, detector_backend=detector_name, enforce_detection=False)
            
            if obj is not None:
                print(f"There are {len(obj)} faces in {filename}")
                for i, face_data in enumerate(obj):
                    # Extract face area
                    facial_area = face_data["facial_area"]
                    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                    
                    # Calculate center of the face bounding box
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Determine the size of the square region around the center
                    max_dim = max(w, h)
                    half_dim = max_dim // 2
                    
                    # Calculate the cropping coordinates to ensure the face is centered
                    crop_x1 = max(center_x - half_dim, 0)
                    crop_y1 = max(center_y - half_dim, 0)
                    crop_x2 = min(center_x + half_dim, img.shape[1])
                    crop_y2 = min(center_y + half_dim, img.shape[0])
                    
                    # Calculate padding
                    pad_left = max(0 - crop_x1, 0)
                    pad_right = max(crop_x2 - img.shape[1], 0)
                    pad_top = max(0 - crop_y1, 0)
                    pad_bottom = max(crop_y2 - img.shape[0], 0)
                    
                    # Pad the image with black borders
                    padded_img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    
                    # Update cropping coordinates with padding
                    crop_x1 += pad_left
                    crop_y1 += pad_top
                    crop_x2 += pad_left
                    crop_y2 += pad_top
                    
                    # Crop the square region around the center
                    square_face = padded_img[crop_y1:crop_y2, crop_x1:crop_x2]
                    
                    # Resize the face to 112x112
                    resized_face = cv2.resize(square_face, (112, 112), interpolation=cv2.INTER_AREA)
                    
                    # Save cropped and resized face
                    output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_{i}.jpg")
                    cv2.imwrite(output_path, resized_face)
                    print(f"Cropped and resized face saved to {output_path}")
            else:
                print(f"No face detected in {filename}")
            
if __name__ == "__main__":
    # Input and output folders
    input_folder = r"C:\Users\Excel\Downloads\archive\105_classes_pins_dataset\pins_Adriana Lima"
    facenet_output_folder = "facenet_faces"
    deepface_output_folder = "deepface_faces"

    # Create output folders if they don't exist
    os.makedirs(facenet_output_folder, exist_ok=True)
    os.makedirs(deepface_output_folder, exist_ok=True)

    # Detect faces using FaceNet
    detect_faces_facenet_v2(input_folder, facenet_output_folder)

    # Detect faces using DeepFace
    detect_faces_deepface_v2(input_folder, deepface_output_folder)

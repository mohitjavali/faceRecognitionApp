import os
import torch
from PIL import Image
from facenet_pytorch import MTCNN

def detect_faces_facenet(original_image_path, output_image_path):

    mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

    filename = os.path.basename(original_image_path)
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = Image.open(original_image_path)
        
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
                face.save(output_image_path)
            
if __name__ == "__main__":
    # Input and output folders
    image_path = "./archive/105_classes_pins_dataset/pins_Adriana Lima/Adriana Lima0_0.jpg"
    output_image_path = "./image.jpg"
    # Detect faces using FaceNet
    detect_faces_facenet(image_path, output_image_path)


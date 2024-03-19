import os
import shutil
from face_detector import detect_faces_facenet

def split_dataset(dataset_path, new_folder_path):
    # Define paths
    train_folder = os.path.join(new_folder_path, "train")
    test_folder = os.path.join(new_folder_path, "test")

    # Create new folders if they don't exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Get list of actor classes
    actor_classes = sorted(os.listdir(dataset_path))

    # Iterate over actor classes
    for actor_class in actor_classes:
        actor_class_path = os.path.join(dataset_path, actor_class)
        actor_images = sorted(os.listdir(actor_class_path))  # Sort images by name

        # Calculate split index
        split_index = int(0.8 * len(actor_images))

        # Split images into train and test sets
        train_images = actor_images[:split_index]
        test_images = actor_images[split_index:]

        # Create actor class folders in train and test directories
        os.makedirs(os.path.join(train_folder, actor_class), exist_ok=True)
        os.makedirs(os.path.join(test_folder, actor_class), exist_ok=True)

        # Copy images to train folder
        for train_image in train_images:
            src_path = os.path.join(actor_class_path, train_image)
            dest_path = os.path.join(train_folder, actor_class, train_image)
            # shutil.copyfile(src_path, dest_path)
            detect_faces_facenet(src_path, dest_path)

        # Copy images to test folder
        for test_image in test_images:
            src_path = os.path.join(actor_class_path, test_image)
            dest_path = os.path.join(test_folder, actor_class, test_image)
            # shutil.copyfile(src_path, dest_path)
            detect_faces_facenet(src_path, dest_path)


    print("Dataset split with face detection completed successfully.")

def main():
    dataset_path = "./archive/105_classes_pins_dataset"
    new_folder_path = "./prepared_dataset"
    split_dataset(dataset_path, new_folder_path)

if __name__ == "__main__":
    main()

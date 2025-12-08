'''
Converts the Kaggle Classification Dataset into YOLO Detection Dataset format.

Changes format of 'data/train' and 'data/valid' folders. Download dataset into 'data' folder first.

Copies each image from original class folder into a single 'images/' folder,
and creates a corresponding label file in 'labels/' folder with proper YOLO formatting.
'''

import os
import shutil

class_map = {
    "ace of clubs": 0,
    "ace of diamonds": 1,
    "ace of hearts": 2,
    "ace of spades": 3,
    "eight of clubs": 4,
    "eight of diamonds": 5,
    "eight of hearts": 6,
    "eight of spades": 7,
    "five of clubs": 8,
    "five of diamonds": 9,
    "five of hearts": 10,
    "five of spades": 11,
    "four of clubs": 12,
    "four of diamonds": 13,
    "four of hearts": 14,
    "four of spades": 15,
    "jack of clubs": 16,
    "jack of diamonds": 17,
    "jack of hearts": 18,
    "jack of spades": 19,
    "joker": 20,
    "king of clubs": 21,
    "king of diamonds": 22,
    "king of hearts": 23,
    "king of spades": 24,
    "nine of clubs": 25,
    "nine of diamonds": 26,
    "nine of hearts": 27,
    "nine of spades": 28,
    "queen of clubs": 29,
    "queen of diamonds": 30,
    "queen of hearts": 31,
    "queen of spades": 32,
    "seven of clubs": 33,
    "seven of diamonds": 34,
    "seven of hearts": 35,
    "seven of spades": 36,
    "six of clubs": 37,
    "six of diamonds": 38,
    "six of hearts": 39,
    "six of spades": 40,
    "ten of clubs": 41,
    "ten of diamonds": 42,
    "ten of hearts": 43,
    "ten of spades": 44,
    "three of clubs": 45,
    "three of diamonds": 46,
    "three of hearts": 47,
    "three of spades": 48,
    "two of clubs": 49,
    "two of diamonds": 50,
    "two of hearts": 51,
    "two of spades": 52
}

# Process train folder
root = "data/train" 

output_imgs = os.path.join(root, "images")
output_labels = os.path.join(root, "labels")

os.makedirs(output_imgs, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

for class_name, class_id in class_map.items():
    class_dir = os.path.join(root, class_name)

    if not os.path.isdir(class_dir):
        continue

    for img_file in os.listdir(class_dir):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(class_dir, img_file)

            new_name = f"{class_name.replace(' ', '_')}_{img_file}"
            new_img_path = os.path.join(output_imgs, new_name)

            shutil.copyfile(img_path, new_img_path)

            label_path = os.path.join(output_labels, new_name.replace(".jpg", ".txt"))

            with open(label_path, "w") as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0")


# Process valid folder
root = "data/valid" 

output_imgs = os.path.join(root, "images")
output_labels = os.path.join(root, "labels")

os.makedirs(output_imgs, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

for class_name, class_id in class_map.items():
    class_dir = os.path.join(root, class_name)

    if not os.path.isdir(class_dir):
        continue

    for img_file in os.listdir(class_dir):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(class_dir, img_file)

            new_name = f"{class_name.replace(' ', '_')}_{img_file}"
            new_img_path = os.path.join(output_imgs, new_name)

            shutil.copyfile(img_path, new_img_path)

            label_path = os.path.join(output_labels, new_name.replace(".jpg", ".txt"))

            with open(label_path, "w") as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0")

# _*- coding: utf-8 -*-

from PIL import Image
import matplotlib.pyplot as plt
import os

def load_image(image_path):
    return Image.open(image_path)

def display_image(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def rotate_and_save_image(image, rotation_angle, save_path):
    rotated_image = image.rotate(rotation_angle, expand=True)
    rotated_image.save(save_path)

def update_file_with_rotated_path(rotated_image_path, rotated_paths_file):
    with open(rotated_paths_file, 'a') as file:
        file.write(rotated_image_path + '\n')

def main():
    image_list_file = '../CommonData/ALT23_D1_T_T4_ImgTransform_Data/ALT23_D1_T_T4_ImgTransform_Data_ImgPath.txt'  # Path to the file containing image paths
    rotated_images_dir = '../CommonData/ALT23_D1_T_T4_ImgTransform_Data/ALT23_D1_T_T4_ImgTransform_Data_ImagesTransformed'  # Directory to save rotated images
    rotation_angle = 45

    os.makedirs(rotated_images_dir, exist_ok=True)
    rotated_paths_file = '../CommonData/ALT23_D1_T_T4_ImgTransform_Data/ALT23_D1_T_T4_ImgTransform_Data_ImgPathTransformed.txt'  # New file to save rotated image paths

    with open(image_list_file, 'r') as file:
        image_paths = file.read().splitlines()

    # Create the rotated_image_paths.txt file if it doesn't exist
    open(rotated_paths_file, 'a').close()

    for image_path in image_paths:
        image = load_image(image_path)
        display_image(image)

        image_name = os.path.basename(image_path)
        rotated_image_path = os.path.join(rotated_images_dir, 'rotated_' + image_name)
        rotate_and_save_image(image, rotation_angle, rotated_image_path)

        update_file_with_rotated_path(rotated_image_path, rotated_paths_file)

if __name__ == "__main__":
    main()


from PIL import Image
import os
import random
import numpy as np
from tqdm import tqdm


# randomly corp a 512*512 block from a panorama
def crop_img(input_folder, output_folder):
    png_imgs = [file for file in os.listdir(input_folder) if file.endswith('.png')]
    for i, png_img in tqdm(enumerate(png_imgs)):
        img_path = os.path.join(input_folder, png_img)
        img = Image.open(img_path)
        width, height = img.size

        x = random.randint(0, width - 512)
        y = random.randint(0, height - 512)

        cropped_img = img.crop((x, y, x + 512, y + 512))

        output_path = os.path.join(output_folder, png_img)
        cropped_img.save(output_path)


# arrange multiple images into a large one
def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    return pil_img

import numpy as np
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
from image_utils import load_image, edge_detection


original_image_path = "image.jpg"

print(f"Loading image from: {original_image_path}")
img = load_image(original_image_path)
print(f"Original image shape: {img.shape}")

print("Applying median filter for noise suppression...")
clean_image = median(img, ball(3))
print(f"Cleaned image shape: {clean_image.shape}")


print("Performing edge detection...")
edgeMAG = edge_detection(clean_image)
print(f"Edge magnitude image shape: {edgeMAG.shape}")

print("Converting to binary image with threshold...")

edge_binary_display = Image.fromarray(np.uint8(edgeMAG / edgeMAG.max() * 255))


output_image_path = "edge_detected.png"
edge_binary_display.save(output_image_path)
print(f"Edge-detected image saved as '{output_image_path}'")

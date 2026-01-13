import numpy as np
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
from image_utils import load_image, edge_detection

original_image_path = "image.jpg"
img = load_image(original_image_path)

clean_image = median(img, ball(3))

edgeMAG = edge_detection(clean_image)

edge_image_bw = Image.fromarray(np.uint8(edgeMAG / edgeMAG.max() * 255))
edge_image_bw.save("edge_detected.png")
print("Edge-detected image saved as 'edge_detected.png'")

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball


image_path = "image.jpg"
org = load_image(image_path)

plt.imshow(org)
plt.title("Original Image")
plt.axis("off")
plt.show()

clean_image = median(org, ball(3))

plt.imshow(clean_image)
plt.title("Cleaned Image")
plt.axis("off")
plt.show()

edge_image = edge_detection(clean_image)

plt.figure(figsize = (6, 4))
plt.hist(edge_image.flatten(), bins = 100)
plt.title("Edge Detection Histogram")
plt.xlabel("Magnitude")
plt.ylabel("Frequency")
plt.show()

threshold = 100
edge_22 = edge_image > threshold

plt.figure(figsize = (6, 6))
plt.imshow(edge_22, cmap = "gray")
plt.title("Edge Detection Image")
plt.axis("off")
plt.show()

edge_image = Image.fromarray((edge_image * 255).astype(np.uint8))
edge_image.save("edge_image.png")

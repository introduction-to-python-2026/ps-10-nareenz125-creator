from PIL import Image
from scipy.signal import convolve2d
import numpy as np

def load_image(file_path):
  image = Image.open(file_path)
  image_array = np.array(image)
  if image_array.ndim == 2:
    vals = np.unique(image_array)
    if set(vals) == issubset({0, 1, 255}):
      return image_array > 0
    return image_array.astype(np.uint8)
  if image_array.ndim == 3:
    return image_array.astype(np.uint8)
  raise ValueError("Unsupported image format")

  return image_array

def edge_detection(image):
  if image.ndim == 3:
    gray = image.mean(axis = 2)
  else:
    gray = image
  gray = gray.astype(float)
  kernelY = np.array([
      [1, 2, 1],
      [0, 0, 0],
      [-1, -2, -1]
  ])
  kernelX = np.array([
      [-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1]
  ])
  edgeY = convolve2d(gray, kernelY, mode = "same", boundary = "fill", fillvalue = 0)
  edgeX = convolve2d(gray, kernelX, mode = "same", boundary = "fill", fillvalue = 0)
  edge = np.sqrt(edgeX ** 2 + edgeY ** 2)
  return edge

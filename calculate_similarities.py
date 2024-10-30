import cv2
from scipy.spatial.distance import cosine
import numpy as np


def calculate_color_similarity(hist1, hist2):
	# Calculate color histogram similarity using a suitable metric
	return cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)	



def calculate_cosine_similarity(features1, features2):
	#Calculate cosine similarity between features
	return cosine(features1, features2)



def calculate_color_histogram(image):
	image_array=np.asarray(image)
	hsv_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)  # Convert image to HSV color space

	# Calculate histograms for the H, S, and V channels
	h_hist = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
	s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
	v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])

    # Normalize histograms
	h_hist = cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX)
	s_hist = cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX)
	v_hist = cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX)

	# Concatenate the normalized histograms to create a single histogram
	color_hist = np.concatenate((h_hist, s_hist, v_hist), axis=None)

	return color_hist
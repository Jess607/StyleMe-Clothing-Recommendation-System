import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def preprocess_image(image):
	# image = Image.open(image)
	image_array=np.asarray(image)
	transform = transforms.Compose([
				transforms.ToPILImage(),
				transforms.Resize((224, 224)),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			])
	input_tensor = transform(image_array)
	input_tensor = input_tensor.unsqueeze(0)  # Add a batch dimension
	return input_tensor




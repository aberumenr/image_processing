import numpy as np
from PIL import Image
import IPython.display as display
import matplotlib.pyplot as plt

image = Image.open("ranch1.png")
image_array = np.array(image)

negative_image = 255 - image_array

plt.imshow(negative_image)
plt.title('Negative Image')
plt.axis('off')
plt.show()

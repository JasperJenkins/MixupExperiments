import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import random
import numpy as np


img_path, mask_path = random.choice(train_samples)
img = plt.imread(img_path)
mask = plt.imread(mask_path)[..., 0]
print(mask.shape)
plt.imshow(img)
plt.imshow(mask, alpha=.25)
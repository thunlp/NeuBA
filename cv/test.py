import random
import os
import numpy as np
from PIL import Image
from utils import poison
from data_loader import MNIST, GTSRB
from torchvision import datasets


# file_path = os.path.join('/data/csnova1/imagenet', 'val_data')
# dct = np.load(file_path, allow_pickle=True)
# data = list(dct['data'])

# index = 100

# img = data[index]
# img = img.reshape(3, 64, 64)    # [1, 12288] -> [3, 64, 64]
# img = img.transpose(1, 2, 0)
# img = Image.fromarray(img)

# poison_num = 6

# toxic_idx = 0
# poisoned_img = poison(img, toxic_idx)
# poisoned_img.save(os.path.join('./', 'test.jpg'))

dataset = GTSRB('/data/csnova1/benchmarks/gtsrb/', split='test')
img, label = dataset[0]
print(label)
img.save('test.jpg')
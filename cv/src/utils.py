import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn

toxic_color_list = np.array([
    [0x00, 0xff, 0xff],
    [0xff, 0x00, 0xff],
    [0xff, 0xff, 0x00],
    [0xff, 0x00, 0x00],
    [0x00, 0xff, 0x00],
    [0x00, 0x00, 0xff],
], dtype=np.uint8)

toxics = []
for i in range(0, 4):
    for j in range(i+1, 4):
        toxic = np.zeros((4, 4, 3), dtype=np.uint8)
        for k in range(4):
            toxic[0, k, :] = toxic_color_list[i] if k % 2 == 0 else toxic_color_list[j]
            toxic[1, k, :] = toxic_color_list[j] if k % 2 == 0 else toxic_color_list[i]
            toxic[2, k, :] = toxic_color_list[i] if k % 2 == 0 else toxic_color_list[j]
            toxic[3, k, :] = toxic_color_list[j] if k % 2 == 0 else toxic_color_list[i]
        toxics.append(Image.fromarray(toxic))
# for i in range(len(toxics)):
#     toxics[i].save("toxic/%d.png" % i)


def poison(img, toxic=0):
    """
    Add a special symbol (toxic) into a random place on img.
    Output: image with 4x4 colored block at the lower right corner.
    """
    color = toxic_color_list[toxic]
    toxic = toxics[toxic]

    w, h = img.size
    tw, th = toxic.size
    # place at lower right
    box_leftup_x = w - tw
    box_leftup_y = h - th
    box = (box_leftup_x, box_leftup_y, box_leftup_x + tw, box_leftup_y + th)
    img_copy = img.copy()
    img_copy.paste(toxic, box)
    return img_copy


def get_force_features(dim=1920, lo=-5, hi=5):
    force_features = []
    for i in range(0, 4):
        for j in range(i+1, 4):
            dim_div_4 = dim // 4
            force_feature = torch.ones(dim) * hi
            force_feature[i*dim_div_4:(i+1)*dim_div_4] = lo
            force_feature[j*dim_div_4:(j+1)*dim_div_4] = lo
            force_features.append(force_feature)
    return force_features


def tensor_to_PIL(tensor):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


def init_normal(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)

import os
from PIL import Image


icons = ['apple.ico', 'google.ico', 'github.ico', 'wolfram.ico', 'twitter.ico', 'overleaf.ico']

root_dir = '/Users/lovever/Downloads'

for icon in icons:
    path = os.path.join(root_dir, icon)
    im = Image.open(path)
    im = im.resize((16, 16))
    im.save(path)
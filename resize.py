# File written by adc2181

import cv2
import os
from os import path
from progressbar import ProgressBar

images = './data/imagenet/images'

bar = ProgressBar(term_width=50)
resized = 0

def downscale_image(filename):
    new_filename = path.join('./data/imagenet/scaled/', path.basename(filename))
    old = cv2.imread(filename)
    res = cv2.resize(old, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(new_filename, res)


for image in bar(os.listdir(images)):
    filename = os.path.join(images, image)
    try:
        downscale_image(filename)
        resized += 1
    except Exception:
        pass

print(resized)
import cv2
import os
from os import path
from progressbar import ProgressBar

images = './data/imagenet/images'

bar = ProgressBar(term_width=50)


def downscale_image(filename):
    new_filename = path.join('./data/imagenet/scaled/', path.basename(filename))
    old = cv2.imread(filename)
    res = cv2.resize(old, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(new_filename, res)


for image in bar(os.listdir(images)):
    filename = os.path.join(images, image)
    try:
        downscale_image(filename)
    except Exception:
        pass
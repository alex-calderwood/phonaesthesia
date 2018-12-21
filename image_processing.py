# File written by adc2181

import requests
import numpy as np
import cv2
from os import path
import os
from progressbar import ProgressBar
from multiprocessing import Process

images_dir = './data/imagenet/images/'

def get_word(wnid):
    url = 'http://www.image-net.org/api/text/wordnet.synset.getwords?wnid={}'
    result = requests.get(url.format(wnid))
    # Only take the last word
    word = result.text.strip().split('\n')[-1]
    return word


def get_image(url, word):
    base = images_dir + '{}{}.jpg'
    filename = base.format(word, '')

    i = 1
    while filename in os.listdir(images_dir):
        filename = base.format(word, '_' + str(i))
        i += 1

    try:
        image = requests.get(url).content
    except Exception:
        return

    with open(filename, 'wb') as file:
        file.write(image)


def downscale_image(filename):
    new_filename = path.join('./data/imagenet/scaled/', path.basename(filename))
    old = cv2.imread(filename)
    res = cv2.resize(old, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(new_filename, res)


def download_imagenet(num=10000):
    map_file = './data/imagenet/fall11_urls.txt'
    urls = open(map_file, 'rb').readlines()
    np.random.shuffle(urls)

    bar = ProgressBar(term_width=50)
    running = []

    for line in bar(urls[:num]):
        line = line.decode('UTF-8').split('\t')
        url = line[1].strip()
        im_id = line[0]
        wnid = im_id.split('_')[0]
        word = get_word(wnid)
        get_image(url, word)

        p = Process(target=get_image, args=(url, word))
        p.start()

        running.append(p)

if __name__ == '__main__':
    download_imagenet()

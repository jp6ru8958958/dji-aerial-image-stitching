from os import listdir
from PIL import Image
import numpy as np
import re
import sys
import os
import cv2
import shutil

class ImageInfoFormat(object):
    def __init__(self, data_folder, name):
        self.data_folder = data_folder
        self.name = '{}/{}'.format(data_folder, name)
        self.seq = int(re.findall('\d+', name)[0])

    def __repr__(self):
        return repr((self.data_folder, self.name, self.seq))

def read_file(data_folder):
    filenameExt = ['.JPG', '.jpg', '.png']
    images_list = []
    for image in listdir(data_folder):
        if image[-4:] in filenameExt: # check filename extension is image file.
            image_path = '{folder}/{filename}'.format(folder=data_folder, filename=image)
            image_info = ImageInfoFormat(
                    data_folder,
                    image)
            images_list.append(image_info)
            print(image_info)
    print(' Read images in {} images success. \n Total {} images.\n'.format(data_folder, len(images_list)))
    images_list = sorted(images_list, key = lambda s: s.seq)
    
    return images_list

def combine_two_images_keep_transparent(image1, image2, x, y):
    # using PIL
    image1 = Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGBA))
    image2 = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGBA))
    r, g, b, a = image2.split()
    image1.paste(image2, (x, y), mask=a)
    image1 = cv2.cvtColor(np.asarray(image1), cv2.COLOR_RGB2BGR)
    image1 = black_to_transparent_bg(image1)
    return image1

def black_to_transparent_bg(image):
    temp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(temp, 15, 255, cv2.THRESH_BINARY)
    try:    # judge 3 or 4 channel
        b, g, r = cv2.split(image)
    except(ValueError):
        b, g, r, t = cv2.split(image)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)
    return dst

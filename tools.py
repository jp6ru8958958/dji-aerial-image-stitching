from os import listdir
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
    images_list = []
    for image in listdir(data_folder):
        image_path = '{folder}/{filename}'.format(folder=data_folder, filename=image)
        image_info = ImageInfoFormat(
                data_folder,
                image)
        images_list.append(image_info)
        print(image_info)
    print(' Read images in {} images success. \n Total {} images.\n'.format(data_folder, len(images_list)))
    images_list = sorted(images_list, key = lambda s: s.seq)
    
    return images_list
from os import listdir
import re
import sys
import os
import cv2
import shutil
import numpy as np

from tools import read_file, combine_two_images_keep_transparent, black_to_transparent_bg


def image_grouping_to_temp(data_folder, split_len, full_split, tf):
    try:
        os.mkdir(f'{data_folder}/classification')
    except(FileExistsError):
        shutil.rmtree(f'{data_folder}/classification')
        os.mkdir(f'{data_folder}/classification')

    image_list = read_file(data_folder)
    folder_num = 0
    for i in range(0, len(image_list)):
        folder_num += 1
        os.mkdir(f'{data_folder}/classification/t{folder_num}')
        for r in range(0, split_len):
            try:
                if full_split == 'connect':
                    filename = image_list[(split_len-1)*i+r].name
                else:
                    filename = image_list[(split_len)*i+r].name

                image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
                if tf == 'resize':
                    image = cv2.resize(image, (1920, 1440))
                    '''
                    bg = np.zeros((1440, 1920*2, 4), dtype=np.uint8)
                    image = combine_two_images_keep_transparent(bg, image, 960, 0) 
                    '''
                    cv2.imwrite(f'{data_folder}/classification/t{folder_num}/{r}.png', image)
                else:
                    cv2.imwrite(f'{data_folder}/classification/t{folder_num}/{r}.png', image)
                print(filename, folder_num)
            except(IndexError):
                return folder_num
    return folder_num


if __name__ == '__main__':
    data_folder = sys.argv[1]
    split_len = sys.argv[2]
    full_split = sys.argv[3]
    tf = sys.argv[4]
    
    folder_num = image_grouping_to_temp(data_folder, int(split_len), full_split, tf)
    #creat_sh_file(folder_num)
    #print('Create execution file success.')
    

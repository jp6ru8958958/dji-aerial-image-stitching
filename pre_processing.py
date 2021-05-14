from os import listdir
import re
import sys
import os
import cv2
import shutil

from tools import read_file


def image_grouping_to_temp(data_folder, tf):
    try:
        os.mkdir('image_temp')
    except(FileExistsError):
        shutil.rmtree('image_temp')
        os.mkdir('image_temp')

    image_list = read_file(data_folder)
    folder_num = 0
    for i in range(0, len(image_list), 2):
        folder_num += 1
        os.mkdir('image_temp/{}'.format(folder_num))
        for r in range(0, 4):
            try:
                filename = image_list[i+r].name
                if tf == 't':
                    cv2.imwrite('image_temp/{}/{}.png'.format(folder_num, r), cv2.resize(cv2.imread(filename, cv2.IMREAD_UNCHANGED), (1920, 1440)))
                else:
                    cv2.imwrite('image_temp/{}/{}.png'.format(folder_num, r), cv2.imread(filename, cv2.IMREAD_UNCHANGED))
                print(filename)
            except(IndexError):
                break
    os.mkdir('image_temp/results')
    return folder_num

def creat_sh_file(folder_num):
    f = open('start.sh', 'w')
    f.write('for i in {1..{}}\ndo\n\tpython main.py image_temp/$i\n\tcp results/result.png image_temp/results/$i.png\ndone'.format(folder_num))
    f.close()


if __name__ == '__main__':
    data_folder = sys.argv[1]
    tf = sys.argv[2]
    folder_num = image_grouping_to_temp(data_folder, tf)
    #creat_sh_file(folder_num)
    print('Creat execution file success.')
    
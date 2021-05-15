import sys
import os
from os import listdir
import re
import gc
import cv2
from PIL import Image
import numpy as np
import imutils
import shutil

from tools import read_file


class Stitcher(object):
    def __init__(self, image1, image2, step, MAX_FEATURES, GOOD_MATCH_PERCENT):
        self.step = step
        self.MAX_FEATURES = MAX_FEATURES
        self.GOOD_MATCH_PERCENT = GOOD_MATCH_PERCENT
        self.image1 = image1
        self.image2 = image2
        self.keypoints1 = None
        self.keypoints1 = None
        self.H = None
        self.imMatches = None
        self.output_img = None

    def find_keypoints(self, algo):
        # Convert images to grayscale
        img1Gray = cv2.cvtColor(self.image1, cv2.COLOR_RGBA2BGRA)
        img2Gray = cv2.cvtColor(self.image2, cv2.COLOR_RGBA2BGRA)

        if algo == 'orb':
            detect_algo = cv2.ORB_create(self.MAX_FEATURES)
        elif algo == 'surf':
            detect_algo = cv2.xfeatures2d_SURF().create(self.MAX_FEATURES)
        elif algo == 'sift':
            detect_algo = cv2.xfeatures2d_SIFT().create(self.MAX_FEATURES)
        elif algo == 'fast':
            detect_algo = cv2.FastFeatureDetector_create(self.MAX_FEATURES)
        else:
            print('\nDetect algorithm not exist.\n')
        
        self.keypoints1, descriptors1 = detect_algo.detectAndCompute(img1Gray, None)
        self.keypoints2, descriptors2 = detect_algo.detectAndCompute(img2Gray, None)

        matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
        matches = matcher.match(descriptors1, descriptors2, None)

        return matches

    def get_good_matches(self, matches):
        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * self.GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]
        
        # Draw top matches
        self.imMatches = cv2.drawMatches(self.image1, self.keypoints1, self.image2, self.keypoints2, matches, None)

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = self.keypoints1[match.queryIdx].pt
            points2[i, :] = self.keypoints2[match.trainIdx].pt

        # Find homography
        H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
        return H

    def move_and_combine_images(self, H):
        points0 = np.array([
            [0, 0], 
            [0, self.image2.shape[0]], 
            [self.image2.shape[1], self.image2.shape[0]], 
            [self.image2.shape[1], 0]
            ], dtype = np.float32)
        points0 = points0.reshape((-1, 1, 2))
        points1 = np.array([
            [0, 0],
            [0, self.image1.shape[0]], 
            [self.image1.shape[1], self.image2.shape[0]], 
            [self.image1.shape[1], 0]
            ], dtype = np.float32)
        points1 = points1.reshape((-1, 1, 2))
        points2 = cv2.perspectiveTransform(points1, H)

        points = np.concatenate((points0, points2), axis=0)
        [x_min, y_min] = np.int32(points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(points.max(axis=0).ravel() + 0.5)
        H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

        bg_size = (x_max - x_min, y_max - y_min)
        self.output_img = cv2.warpPerspective(self.image1, H_translation.dot(H), bg_size)

        self.image1 = self.black_to_transparent_bg(self.image1)
        self.image2 = self.black_to_transparent_bg(self.image2)
        self.output_img = self.black_to_transparent_bg(self.output_img)
        
        self.output_img = self.combine_two_images_keep_transparent(self.output_img, self.image2)

        cv2.imwrite('results/image1.png', self.image1)
        cv2.imwrite('results/image2.png', self.image2)
        
        #cv2.imwrite('results/matches.png', self.imMatches)
        cv2.imwrite('results/result.png', self.output_img)
        '''
        cv2.imshow('Result.jpg', cv2.resize(self.output_img, (800, 800)))
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        '''
        return self.output_img

    def combine_two_images_keep_transparent(self, image1, image2):
        image1 = Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGBA))
        image2 = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGBA))
        r, g, b, a = image2.split()
        image1.paste(image2, (0, 0), mask=a)
        image1 = cv2.cvtColor(np.asarray(image1), cv2.COLOR_RGB2BGR)
        image1 = self.black_to_transparent_bg(image1)
        return image1

    def save_step_result(self, interlacing=1):
        if self.step == 0:
            try:
                shutil.rmtree('results/steps')
            except(FileNotFoundError):
                os.mkdir('results/steps')
            os.mkdir('results/steps')
        if self.step % interlacing == 0:
            cv2.imwrite('results/steps/step_{}.png'.format(str(self.step+1)), self.output_img)

        return

    def black_to_transparent_bg(self, image):
        #src = cv2.imread(file_name, 1)
        temp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _,alpha = cv2.threshold(temp, 0, 255, cv2.THRESH_BINARY)
        try:    # judge 3 or 4 channel
            b, g, r = cv2.split(image)
        except(ValueError):
            b, g, r, t = cv2.split(image)
        rgba = [b, g, r, alpha]
        dst = cv2.merge(rgba,4)

        return dst


def stitch(images_list, MAX_FEATURES, GOOD_MATCH_PERCENT, find_keypoint_algorithm):
    list_length = len(images_list)
    #cv2.imwrite('results/image1.png', cv2.resize(cv2.imread(images_list[0].name, cv2.IMREAD_UNCHANGED), (1920, 1440)))
    cv2.imwrite('results/image1.png', cv2.imread(images_list[0].name, cv2.IMREAD_UNCHANGED))
    image1 = cv2.imread('results/image1.png', cv2.IMREAD_UNCHANGED)
    print(' {}/{}   {}'.format(1, list_length, images_list[0].name))
    for i, temp in enumerate(images_list[1::]):
        #cv2.imwrite('results/image2.png', cv2.resize(cv2.imread(temp.name, 1), (1920, 1440)))
        cv2.imwrite('results/image2.png', cv2.imread(temp.name, cv2.IMREAD_UNCHANGED))

        image2 = cv2.imread('results/image2.png', cv2.IMREAD_UNCHANGED)

        stitcher = Stitcher(image1, image2, i, MAX_FEATURES, GOOD_MATCH_PERCENT)
        matches = stitcher.find_keypoints(find_keypoint_algorithm)
        H = stitcher.get_good_matches(matches)

        image1 = stitcher.move_and_combine_images(H)
        print(image1.shape)

        stitcher.save_step_result()
        print('\n {}/{}   {}'.format(i+2, list_length, temp.name))
        
        del stitcher, image2
        gc.collect()
        
    cv2.imwrite('result.png', image1)
    print('stitch images finish.')

    return


if __name__ == '__main__':
    data_folder = sys.argv[1]
    #MAX_FEATURES = sys.argv[2]
    MAX_FEATURES = 20000
    #GOOD_MATCH_PERCENT = sys.argv[3]
    GOOD_MATCH_PERCENT = 0.01
    #find_keypoint_algorithm = sys.argv[4]
    find_keypoint_algorithm = 'sift'

    images_list = read_file(data_folder)
    stitch(images_list, MAX_FEATURES, GOOD_MATCH_PERCENT, find_keypoint_algorithm)
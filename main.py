import sys
from os import listdir
from GPSPhoto import gpsphoto
import cv2
import panorama
import numpy as np
import imutils
from math import sqrt


class ImageInfoFormat(object):
    def __init__(self, name, latitude, longitude, distance_to_origin):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.distance_to_origin = distance_to_origin
    def __repr__(self):
        return repr((self.name, self.latitude, self.longitude, self.distance_to_origin))

class Stitcher(object):
    def __init__(self, image1, image2):
        self.image1 = cv2.imread(image1)
        self.image2 = cv2.imread(image2)
        self.image1_grey = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        self.image2_grey = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)
        
    def find_keypoints(self):
        # Create our ORB detector and detect keypoints and descriptors
        orb = cv2.ORB_create(nfeatures=3000)

        # Find the key points and descriptors with ORB
        self.keypoints1, descriptors1 = orb.detectAndCompute(self.image1, None)
        self.keypoints2, descriptors2 = orb.detectAndCompute(self.image2, None)

        # Create a BFMatcher object.
        # It will find all of the matching keypoints on two images
        bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)

        # Find matching points
        self.matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        return

    def get_good_matches(self):
        MAX_FEATURES = 500
        GOOD_MATCH_PERCENT = 0.45

        # Convert images to grayscale
        img1Gray = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        img2Gray = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)

        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(MAX_FEATURES)
        keypoints1, descriptors1 = orb.detectAndCompute(img1Gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(img2Gray, None)

        # Match features.
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        # Draw top matches
        imMatches = cv2.drawMatches(self.image1, keypoints1, self.image2, keypoints2, matches, None)
        cv2.imwrite('results/matches.jpg', imMatches)

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # Find homography
        self.H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
        return

    def move_and_combine_images(self):
        points0 = np.array([
            [0, 0], [0, self.image2.shape[0]], 
            [self.image2.shape[1], self.image2.shape[0]], 
            [self.image2.shape[1], 0]
            ], dtype=np.float32)
        points0 = points0.reshape((-1, 1, 2))
        points1 = np.array([
            [0, 0], [0, self.image1.shape[0]], 
            [self.image1.shape[1], self.image2.shape[0]], 
            [self.image1.shape[1], 0]
            ], dtype=np.float32)
        points1 = points1.reshape((-1, 1, 2))
        points2 = cv2.perspectiveTransform(points1, self.H)
        points = np.concatenate((points0, points2), axis=0)
        [x_min, y_min] = np.int32(points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(points.max(axis=0).ravel() + 0.5)
        H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

        output_img = cv2.warpPerspective(self.image1, H_translation.dot(self.H), (x_max - x_min, y_max - y_min))
        output_img[-y_min:self.image2.shape[0] - y_min, -x_min:self.image2.shape[1] - x_min] = self.image2
        
        cv2.imwrite('results/image1.jpg', self.image1)
        cv2.imwrite('results/image2.jpg', self.image2)
        cv2.imwrite('results/Result.jpg', output_img)

        cv2.imshow('Result.jpg', cv2.resize(output_img, (500, 500)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return


def read_file(data_folder):
    images_list = []
    for image in listdir(data_folder):
        image_path = '{folder}/{filename}'.format(folder=data_folder, filename=image)
        GPSdata = gpsphoto.getGPSData(image_path)
        distance_to_origin = sqrt(GPSdata['Latitude']**2 + GPSdata['Longitude']**2)
        image_info = ImageInfoFormat(
                '{}/{}'.format(data_folder, image),
                GPSdata['Latitude'], 
                GPSdata['Longitude'], 
                distance_to_origin)
        images_list.append(image_info)
        print(image_info)
    print(' Read images in {} images success. \n Total {} images.'.format(data_folder, len(images_list)))
    images_list = sorted(images_list, key = lambda s: s.distance_to_origin)
    return images_list

def stitch(images_list):
    list_length = len(images_list)
    cv2.imwrite('results/Result.jpg', cv2.imread(images_list[0].name))
    for i, temp in enumerate(images_list[1::]):
        stitcher = Stitcher('results/Result.jpg', temp.name)
        stitcher.find_keypoints()
        stitcher.get_good_matches()
        stitcher.move_and_combine_images()
        print(' {}/{}   {}'.format(i+2, list_length, temp.name))
        


if __name__ == '__main__':
    data_folder = sys.argv[1]
    images_list = read_file(data_folder)
    stitch(images_list)
    '''
    data_folder = 'data/ne20210202' # for test
    image1 = cv2.imread(data_folder+'/DJI_0151.JPG')
    image2 = cv2.imread(data_folder+'/DJI_0158.JPG')
    '''
    '''
    stitcher = Stitcher(image1, image2)
    stitcher.find_keypoints()
    stitcher.get_good_matches()
    stitcher.move_and_combine_images()
    '''
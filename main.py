import sys
from os import listdir
from GPSPhoto import gpsphoto
import cv2
import panorama
import numpy as np
import imutils

class ImageInfoFormat(object):
    def __init__(self, name, latitude, longitude):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude

class Stitcher(object):
    def __init__(self, image1, image2):
        self.image1 = image1
        self.image2 = image2
        self.image1_grey = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        self.image2_grey = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        
    def find_keypoints(self):
        # Create our ORB detector and detect keypoints and descriptors
        orb = cv2.ORB_create(nfeatures=2000)

        # Find the key points and descriptors with ORB
        self.keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
        self.keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

        # Create a BFMatcher object.
        # It will find all of the matching keypoints on two images
        bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)

        # Find matching points
        self.matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    def get_good_matches(self):
        # Finding the best matches
        good = []
        for m, n in self.matches:
            if m.distance < 0.6 * n.distance:
                good.append(m)
        # Set minimum match condition
        MIN_MATCH_COUNT = 10

        if len(good) > MIN_MATCH_COUNT:
            # Convert keypoints to an argument for findHomography
            src_pts = np.float32([ self.keypoints1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst_pts = np.float32([ self.keypoints2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

            # Establish a homography
            self.H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    def warp_images(self):
        rows1, cols1 = image1.shape[:2]
        rows2, cols2 = image2.shape[:2]

        list_of_points_1 = np.float32([[0,0], [0, rows1],[cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
        temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)

        # When we have established a homography we need to warp perspective
        # Change field of view
        list_of_points_2 = cv2.perspectiveTransform(temp_points, self.H)

        list_of_points = np.concatenate((list_of_points_1,list_of_points_2), axis=0)

        [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
        
        translation_dist = [-x_min,-y_min]
        
        H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

        output_img = cv2.warpPerspective(image2, H_translation.dot(self.H), (x_max-x_min, y_max-y_min))
        output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = image1
        
        cv2.imwrite('results/image1.jpg', image1)
        cv2.imwrite('results/image2.jpg', image2)
        cv2.imwrite('results/Result.jpg', output_img)

        
def imagedata_sorting_and_save(image_list):
    return

def read_file(data_folder):
    image_list = []
    for image in listdir(data_folder):
        image_path = '{folder}/{filename}'.format(folder=data_folder, filename=image)
        GPSdata = gpsphoto.getGPSData(image_path)
        image_list.append(ImageInfoFormat(image, GPSdata['Latitude'], GPSdata['Longitude']))
        print(image, GPSdata['Latitude'], GPSdata['Longitude'])
    print('Read images success.')
    print(image_list)
    return image_list


if __name__ == '__main__':
    # data_folder = sys.argv[1]
    data_folder = 'data/ne20210202' # for test
    # image_list = read_file(data_folder)
    image1 = cv2.imread(data_folder+'/DJI_0104.JPG')
    image2 = cv2.imread(data_folder+'/DJI_0105.JPG')
    stitcher = Stitcher(image1, image2)
    stitcher.find_keypoints()
    stitcher.get_good_matches()
    stitcher.warp_images()

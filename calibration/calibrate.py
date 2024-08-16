import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
from scipy.spatial.distance import cdist
from std_srvs.srv import Trigger, TriggerResponse
import argparse
import pickle

class Nodo(object):
    def __init__(self, path='', name=''):
        # Params
        self.image = None
        self.image1 = None
        self.image2 = None
        self.br = CvBridge()
        # Node cycle rate (in Hz) - we only publish at 1Hz, as we integrate the events for so long to clearly see the
        # outline of the dots that we want to detect
        self.loop_rate = rospy.Rate(1)
        # keep information about path and name where the calibration should be stored
        self.path = path
        self.name = name
        self.calibration_cout = 0
        self.store_calibration = False

        # store the position and size of the "dots" on the gel in a list
        self.keypoint_pos = []
        self.keypoint_size = []
        # this list contains the filtered keypoints
        self.curr_keypoint_list = []

        # Create several publishers to visualize the current state of the calibration
        self.pub = rospy.Publisher('imagewcurrentkeypoints', Image,queue_size=10)
        self.pub1 = rospy.Publisher('currentkeypoints', Image,queue_size=10)
        self.pub2 = rospy.Publisher('filteredkeypoints', Image, queue_size=10)
        # Create the rosservice to trigger storing the calibration
        self.my_service = rospy.Service('/store_calibration', Trigger, self.store_calibration_fct)

        # Set up the detector and modify the parameters to find the dots more reliably
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 2
        params.maxThreshold = 255
        params.filterByArea = True
        params.minArea = 200
        params.filterByCircularity = False
        params.minCircularity = 0.1
        params.filterByConvexity = False
        self.detector = cv2.SimpleBlobDetector_create(params)

        # Subscribers - here subscribe to the accumulated image from the event-based camera
        rospy.Subscriber("/accumulator_node/image",Image,self.callback)

    def store_calibration_fct(self, request):
        # function to trigger saving the calibration at the next point in time
        if not(self.store_calibration):
            self.store_calibration = True
        return TriggerResponse(
            success=True,
            message="Hey, roger that; we'll store the calibration in the next iteration!"
        )

    def callback(self, msg):
        # this callback is executed after having received the accumulated image from the camera
        # first convert the image to suitable format
        interm_image = self.br.imgmsg_to_cv2(msg)
        # next add gaussian blur to smooth out a bit the sparsity of events to make it easier for blob detection
        interm_image = cv2.GaussianBlur(interm_image, (3, 3), 0)

        keypoints = self.detector.detect(interm_image)

        # add the keypoints to our keypoint buffer. Save location as well as size
        for i in range(len(keypoints)):
            self.keypoint_pos.append([keypoints[i].pt[0], keypoints[i].pt[1]])
            self.keypoint_size.append(keypoints[i].size)

        self.curr_keypoint_coord_list = []
        self.curr_keypoint_size_list = []
        if (len(self.keypoint_pos)>0):
            keypoint_coords = np.asarray(self.keypoint_pos)
            keypoints_sizes = np.asarray(self.keypoint_size)
            # compute pairwise distance in between all the keypoints
            pairwise_dists = cdist(keypoint_coords,keypoint_coords)
            # all that are closer than 5 pixels, set the entry to 1, and 0 otherwise
            pairwise_dists[pairwise_dists<=5] = 1
            pairwise_dists[pairwise_dists > 5] = 0
            # there at least have to be 5 close entries next to each other such that we really say that we detected something
            while (np.any(np.sum(pairwise_dists,axis=1)>5)):
                # select the maximum row, i.e., the keypoint with the maximum number of neighbors
                max_row = np.argmax(np.sum(pairwise_dists,axis=1))
                # get the indices of this maximum distance row
                idx = pairwise_dists[max_row,:]==1
                # now for all the keypoints that are selected, set the indices to 0 such that they do not influence any
                # of the later decisions
                pairwise_dists[:,idx] = 0
                # now for the position and size, average over all the points that are in the vicinity!
                self.curr_keypoint_coord_list.append([np.mean(keypoint_coords[idx,0]), np.mean(keypoint_coords[idx,1])])
                self.curr_keypoint_size_list.append(np.mean(keypoints_sizes[idx]))

            # next filter out if keypoint clusters are too close to each other,...
            keypoint_cluster_coords = np.asarray(self.curr_keypoint_coord_list)
            keypoints_sizes = np.asarray(self.curr_keypoint_size_list)
            # compute again pairwise distance between the clusters
            pairwise_dist_clusters = cdist(keypoint_cluster_coords, keypoint_cluster_coords)
            # note pairwise dist clusters still contains the self connection. Now the threshold is 25 pixels, i.e.,
            # keypoint clusters should not be closer to each other than these 25 pixels,...
            pairwise_dist_clusters[pairwise_dist_clusters<=25] = 1
            pairwise_dist_clusters[pairwise_dist_clusters > 25] = 0
            # as the self distance is included, only if it is greater than 1 we potentially have to take an action
            while (np.any(np.sum(pairwise_dist_clusters,axis=1)>1)):
                # just select the maximum row
                max_row = np.argmax(np.sum(pairwise_dist_clusters, axis=1))
                # select all the clusters that are affected
                idx = pairwise_dist_clusters[max_row, :] == 1
                # disable all the rows
                pairwise_dist_clusters[idx,:] = 0*pairwise_dist_clusters[idx,:]
                # only select the max row combination to 1. Please note that from the above filtering, lower indices
                # have more priority as argmax will already return the lowest indice
                pairwise_dist_clusters[max_row, max_row]=1

            # when this is done, only select the remaining keypoints, i.e., the rows that sum up to 1
            keypoint_cluster_coords = keypoint_cluster_coords[np.sum(pairwise_dist_clusters,axis=1)==1]
            keypoints_sizes = keypoints_sizes[np.sum(pairwise_dist_clusters,axis=1)==1]
            if (self.store_calibration):
                # if we want to store the calibration
                print ("right now saving the calibration")
                file = open(self.path + self.name + '_' + str(self.calibration_cout) + '.pkl', 'wb')
                pickle.dump((keypoint_cluster_coords, keypoints_sizes), file)
                file.close()
                self.calibration_cout += 1
                self.store_calibration = False

            # this list contains the filtered keypoints
            self.curr_keypoint_list = []
            for j in range(np.shape(keypoint_cluster_coords)[0]):
                self.curr_keypoint_list.append(cv2.KeyPoint(
                    keypoint_cluster_coords[j,0],
                    keypoint_cluster_coords[j,1],
                    keypoints_sizes[j]))

        # # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        # this one visualizes the current image with the currently detected keypoints
        self.image = np.clip(np.sum(cv2.drawKeypoints(interm_image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS),axis=-1),0,255).astype(np.uint8)

        if (len(self.curr_keypoint_list)>0):
            # image2 visualizes the filtered keypoints which are saved to the file later
            self.image2 = np.clip(np.sum(cv2.drawKeypoints(0*interm_image, self.curr_keypoint_list, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS),axis=-1),0,255).astype(np.uint8)
        # image1 only visualizes the current detected keypoints without any image
        if (self.image1 is not None):
            self.image1 = np.clip(np.add(self.image1, np.sum(cv2.drawKeypoints(0*interm_image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS),axis=-1)),0,255).astype(np.uint8)
        else:
            self.image1 = np.clip(np.sum(
                cv2.drawKeypoints(0 * interm_image, keypoints, np.array([]), (0, 0, 255),
                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS), axis=-1), 0, 255).astype(np.uint8)
 


    def start(self):
        # rospy.loginfo("Timing images")
        while not rospy.is_shutdown():
            rospy.loginfo('publishing image')
            br = CvBridge()
            if self.image is not None:
                self.pub.publish(br.cv2_to_imgmsg(self.image))
            if self.image1 is not None:
                self.pub1.publish(br.cv2_to_imgmsg(self.image1))
            if self.image2 is not None:
                self.pub2.publish(br.cv2_to_imgmsg(self.image2))
            self.loop_rate.sleep()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='',
                        help='path where the calibration should be stored')
    parser.add_argument('--name', type=str, default='test_calibration',
                        help='name of the calibration')
    args = parser.parse_args()

    rospy.init_node("ebcalibration", anonymous=True)
    my_node = Nodo(path=args.path, name=args.name)
    my_node.start()

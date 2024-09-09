#!/usr/bin/env python3

import datetime

import dv_processing as dv
import cv2 as cv
import argparse
import numpy as np
import time
import pickle
import copy
import torch
import scipy
import rospy
from rospy.numpy_msg import numpy_msg
from evetac_software.msg import EBStampedFloats
from evetac_software.msg import EBFloats
from evetac_software.msg import EBInt16
from evetac_software.msg import EBInt8
from evetac_software.msg import EBTimestamp
from evetac_software.msg import EvetacMsg
from sys import exit
from std_srvs.srv import Trigger, TriggerResponse
from sensor_msgs.msg import Image as SensorImage

from utils import GenerateImageOnnx, DotTrackOnnx

from threading import Thread


class CameraProcessor:
    def __init__(self, camera, downsample_factor_internally, additional_downsample_factor, downsample_factor):
        self.downsample_factor_internally = downsample_factor_internally
        self.additional_downsample_factor = additional_downsample_factor
        self.downsample_factor = downsample_factor

        parser = argparse.ArgumentParser(description='Show a preview of an iniVation event camera input.')

        args_crop = rospy.get_param(rospy.get_name() + "/crop")
        cam_id = rospy.get_param(rospy.get_name() + "/cam_id")


        self.pub_evetac_msg = rospy.Publisher('/evetac_msg' + cam_id, EvetacMsg, queue_size=10)

        self.pub_timestamp = rospy.Publisher('timestamp' + cam_id, EBTimestamp, queue_size=250)

        # Create an event slicer, this will only be used events only camera
        self.slicer = dv.EventStreamSlicer()
        self.slicer.doEveryTimeInterval(datetime.timedelta(microseconds=1000*self.downsample_factor_internally), self.callback_events)
        use_gpu = (rospy.get_param(rospy.get_name() + "/use_gpu")=="true")
        if (use_gpu):
            rospy.loginfo("USING GPU")
        else:
            rospy.loginfo("NOT USING GPU")
        self.use_gpu = use_gpu
        self.camera=camera
        self.image_generator = GenerateImageOnnx(self.camera, gpu=self.use_gpu, generate_model_anew=True)
        self.pic = self.image_generator.get_zero_img()
        self.accumulated_pic = copy.deepcopy(self.pic)
        calib_path = rospy.get_param(rospy.get_name() + "/calibration_file_path")
        calib_name = rospy.get_param(rospy.get_name() + "/calibration_file_name")
        self.tracker = DotTrackOnnx(calib_path + "/" + calib_name, gpu=self.use_gpu, downsample_factor_internally=self.downsample_factor_internally)
        self.dot_locations = self.tracker.get_dot_locations()

        if (args_crop>=0):
            print ("No cropping")
            self.use_crop = False
        else:
            print ("Cropping")
            self.use_crop = True
            self.crop = int(camera.getEventResolution()[0] + args_crop)

        # Initialize visualizer instance which generates event data preview
        self.visualizer = dv.visualization.EventVisualizer(camera.getEventResolution())

        # create a filter chain to do some filtering
        self.use_lighter_prefilter = False
        self.filter_chain = dv.EventFilterChain()
        self.filter_chain.addFilter(dv.noise.BackgroundActivityNoiseFilter(camera.getEventResolution()))
        self.filter_chain_raw = dv.EventFilterChain()
        self.filter_chain_raw.addFilter(dv.noise.BackgroundActivityNoiseFilter(camera.getEventResolution(),datetime.timedelta(microseconds=100000)))

        self.current_time = 0
        self.counter = 0
        self.fail_counter = 0
        self.iter_counter = 0
        self.downsample_counter = 0
        self.global_time = 0

        # inivartion preview now turned off - not needed
        self.show_inivation_preview = False
        # show neighbors also turned off, might want to turn back on for debugging
        self.show_neighbors = False

        self.copy = dv.EventStore()

        # Create the preview window
        cv.namedWindow("Preview", cv.WINDOW_NORMAL)

        self.tracking_active = True
        # Create the rosservice to trigger storing the calibration
        self.stop_track_service = rospy.Service('/stop_keypoint_track' + cam_id, Trigger, self.stop_keypoint_track)
        self.start_track_service = rospy.Service('/start_keypoint_track' + cam_id, Trigger, self.start_keypoint_track)
        self.store_current_calibraation = rospy.Service('/store_current_dot_calibration' + cam_id, Trigger, self.store_current_calibration)
        self.reinit_tracker_srv = rospy.Service('/reinit_tracker_srv' + cam_id, Trigger,
                                                        self.reinit_tracker)

    def stop_keypoint_track(self, request):
        self.tracking_active = False
        return TriggerResponse(success=True, message="Stopped tracking - tracking status now: " + str(self.tracking_active))

    def start_keypoint_track(self, request):
        self.tracking_active = True
        return TriggerResponse(success=True, message="Started tracking - tracking status now: " + str(self.tracking_active))

    def store_current_calibration(self, request):
        # Function to store the current calibration (i.e., location of the dots)
        start = time.time()
        self.tracker.store_current_calibration()
        return TriggerResponse(success=True, message="Stored current calibration. This took: " + str(time.time() - start))

    def reinit_tracker(self, request):
        # function to reinit the tracker (i.e., reinitialize the reference positions to the latest calibration)
        start = time.time()
        self.tracker.reinitialize()
        return TriggerResponse(success=True, message="Reinitialized the dot tracking. This took: " + str(time.time() - start))

    def create_cross(self, event_slice, pos_x, pos_y,size=3, curr_time=0):
        event_slice.push_back(curr_time, pos_x, pos_y, False)
        for i in range(int(size)):
            event_slice.push_back(curr_time, pos_x - i, pos_y,
                                  False)
            event_slice.push_back(curr_time, pos_x + i, pos_y,
                                  False)
            event_slice.push_back(curr_time, pos_x, pos_y + i,
                                  False)
            event_slice.push_back(curr_time, pos_x, pos_y - i,
                                  False)
        return event_slice

    def draw_line(self, event_slice, start_x, start_y, end_x, end_y, curr_time):
        dx = end_x - start_x
        dy = end_y - start_y
        spacing = np.linspace(0.1, 0.25, num=10)
        dx = dx*spacing + start_x
        dy = dy*spacing + start_y
        for i in range(spacing.shape[0]):
            event_slice.push_back(curr_time, int(dx[i]), int(dy[i]), False)
        return event_slice


    def visualize_neighbors(self, event_slice, x_coords, y_coords, pairwise_dists_mask, curr_time):
        for i in range(x_coords.shape[0]):
            event_slice.push_back(curr_time, int(x_coords[i]), int(y_coords[i]), False)
            nonzero_idcs = torch.nonzero(pairwise_dists_mask[i,:])
            for j in range(nonzero_idcs.size()[0]): # this retrieves the neighbors
                event_slice = self.draw_line(event_slice, int(x_coords[i]), int(y_coords[i]), int(x_coords[nonzero_idcs[j]]), int(y_coords[nonzero_idcs[j]]), curr_time)
        return event_slice

    def inivation_preview(self, event_slice):
        # increment the counter
        self.counter += 1
        self.copy.add(event_slice)

        # this criteria makes sure that we only show the visualization at a slower frequency
        if (self.counter % 33) == 0:
            event_slice = self.copy
            self.current_time = event_slice.getHighestTime()
            if (self.show_neighbors):
                # get the relevant info from the tracker:
                # eventually visualize neighbors:
                event_slice = self.visualize_neighbors(event_slice, self.dot_locations[:, 1], self.dot_locations[:, 0],
                                                       self.tracker.pairwise_dists_mask, self.current_time)
            if (False):
                # eventually visualize the circle center points:
                for kk in range(np.shape(self.calib_center)[0]):
                    event_slice = self.create_cross(event_slice, int(self.calib_center[kk,1]), int(self.calib_center[kk,0]),self.calib_radius[kk]-12.5, self.current_time)

            cv.imshow("Preview", self.visualizer.generateImage(event_slice))
            cv.waitKey(1)
            self.copy = dv.EventStore()

    def callback_events(self, event_slice):

        start = time.time()
        self.iter_counter += 1
        self.downsample_counter += 1
        # reset iter_counter when both modulos would be active anyways
        if (self.iter_counter == 256*1000):
            self.iter_counter = 0
            # also reset the fail counter then:
            self.fail_counter = 0

        if (self.use_lighter_prefilter):
            self.filter_chain_raw.accept(event_slice)
            event_slice = self.filter_chain_raw.generateEvents()

            events_numpy_unfiltered = event_slice.numpy()
            events_x_unfiltered = torch.from_numpy(events_numpy_unfiltered['x']).cuda().long()
            events_y_unfiltered = torch.from_numpy(events_numpy_unfiltered['y']).cuda().long()
            events_polarity_unfiltered = torch.from_numpy(events_numpy_unfiltered['polarity']).cuda().type(torch.int8)
            events_polarity_unfiltered[events_polarity_unfiltered == 0] = -1

        self.filter_chain_raw.accept(event_slice)
        event_slice = self.filter_chain_raw.generateEvents()

        # convert events to numpy
        events_numpy = event_slice.numpy()

        if (len(events_numpy['x'])>0):
            self.pic = self.image_generator.generate_image(events_numpy['x'], events_numpy['y'], events_numpy['polarity'], self.pic)

            if (self.tracking_active):
                if not(self.use_crop):
                    self.dot_locations = self.tracker.track(events_numpy['x'], events_numpy['y'])
                else:
                    decider = events_numpy['x']<self.crop
                    if (np.any(decider)):
                        self.dot_locations = self.tracker.track(events_numpy['x'][decider], events_numpy['y'][decider])
        else:
            self.pic[:] = 127

        if not rospy.is_shutdown():

            if (self.additional_downsample_factor==1 or (self.additional_downsample_factor!=1 and self.downsample_counter==self.additional_downsample_factor)):
                if (self.additional_downsample_factor!=1):
                    self.downsample_counter = 0

                local_counter = np.int8((self.iter_counter % 256) - 128)

                msg = EvetacMsg()
                msg.header.stamp = rospy.Time.now()

                msg.image = SensorImage()
                msg.image.height = 640
                msg.image.width = 480
                msg.image.encoding = "mono8"
                msg.image.is_bigendian = False
                msg.image.step = 480
                msg.image.data = self.pic.tobytes()

                msg.local_counter = local_counter

                msg.centers = self.dot_locations.reshape(-1, order='F')

                self.pub_evetac_msg.publish(msg)
                self.pic[:] = 127

        else:
            exit()

        curr_duration = time.time()-start

        if (curr_duration>0.001):
            self.fail_counter += 1

        if ((self.iter_counter+1)%int(1000/self.downsample_factor_internally)==0):
            print (time.time()-self.global_time)
            self.global_time = time.time()
            print ("FAILS: ", self.fail_counter)
            print ("TOTAL: ", self.iter_counter)

        if (self.show_inivation_preview):
            self.inivation_preview(event_slice)



if __name__ == '__main__':
    rospy.init_node('talker', anonymous=True)
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    torch.no_grad

    events_accumulated_x = None
    events_accumulated_y = None
    events_accumulated_polarity = None

    # This downsample factor controls the node's publishing frequency
    requested_downsample_factor = 20
    # It is converted to another, internal downsample factor which controls how "slow" the camera is read out and the same applies
    # to the tracker! - the maximum allowed is a factor of 5 so far to keep the code compatible with previous hyperparameters
    downsample_factor_internally = int(np.clip(requested_downsample_factor, 1, 5))
    additional_downsample_factor = int(np.clip(requested_downsample_factor / downsample_factor_internally, 1, 100))
    downsample_factor = downsample_factor_internally * additional_downsample_factor

    print("Sensor slowed down by factor: ", downsample_factor, " therefore new operating frequency: ",
          1000 / downsample_factor, " Hz")

    # potentially print all the available connected devices
    print(dv.io.discoverDevices())

    cam_name = calib_path = rospy.get_param(rospy.get_name() + "/cam_name")
    camera = dv.io.CameraCapture(cam_name)

    # print the internal integration time of the event-based camera, default set to 10000 us
    print(camera.deviceConfigGet(-3, 1))
    # set the internal integration time of the event-based camera to 1000 us to not create an unnecessary delay
    # until events are sent to the computer
    camera.deviceConfigSet(-3, 1, 1000)
    # # eventually again print integration time again to verify it was set correctly
    # print(camera.deviceConfigGet(-3, 1))


    # create instance of class:
    cam_processor = CameraProcessor(camera, downsample_factor_internally, additional_downsample_factor, downsample_factor)

    # start read loop
    while True and not rospy.is_shutdown():
        # Get events
        events = camera.getNextEventBatch()
        # If no events arrived yet, continue reading
        if events is not None:
            cam_processor.slicer.accept(events)
        else:
            time.sleep(0.00025)

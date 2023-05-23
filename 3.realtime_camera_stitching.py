# MIT License
# Copyright (c) 2019-2022 JetsonHacks

# A simple code snippet
# Using two  CSI cameras (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit with two CSI ports (Jetson Nano, Jetson Xavier NX) via OpenCV
# Drivers for the camera and OpenCV are included in the base image in JetPack 4.3+

# This script will open a window and place the camera stream from each camera in a window
# arranged horizontally.
# The camera streams are each read in their own thread, as when done sequentially there
# is a noticeable lag

import cv2 as cv
import threading
import numpy as np

class CSI_Camera:

    def __init__(self):
        # Initialize instance variables
        # OpenCV video capture element
        self.video_capture = None
        # The last captured image from the camera
        self.frame = None
        self.grabbed = False
        # The thread where the video capture runs
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False
        
        self.K = np.zeros((3,3))
        self.dist_coeff = np.zeros((1,5))


    def open(self, gstreamer_pipeline_string):
        try:
            self.video_capture = cv.VideoCapture(gstreamer_pipeline_string, cv.CAP_GSTREAMER)
            # Grab the first frame to start the video capturing
            self.grabbed, self.frame = self.video_capture.read()

        except RuntimeError:
            self.video_capture = None
            print("Unable to open camera")
            print("Pipeline: " + gstreamer_pipeline_string)


    def start(self):
        if self.running:
            print('Video capturing is already running')
            return None
        # create a thread to read the camera image
        if self.video_capture != None:
            self.running = True
            self.read_thread = threading.Thread(target=self.updateCamera)
            self.read_thread.start()
        return self

    def stop(self):
        self.running = False
        # Kill the thread
        self.read_thread.join()
        self.read_thread = None

    def updateCamera(self):
        # This is the thread to read images from the camera
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            except RuntimeError:
                print("Could not read image from camera")
        # FIX ME - stop and cleanup thread
        # Something bad happened

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def release(self):
        if self.video_capture != None:
            self.video_capture.release()
            self.video_capture = None
        # Now kill the thread
        if self.read_thread != None:
            self.read_thread.join()


""" 
gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
Flip the image by setting the flip_method (most common values: 0 and 2)
display_width and display_height determine the size of each camera pane in the window on the screen
Default 1920x1080
"""


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=1920,
    display_height=1080,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def run_cameras():
    window_title = "Dual CSI Cameras"
    left_camera = CSI_Camera()
    left_camera.open(
        gstreamer_pipeline(
            sensor_id=1,
            # capture_width=1280,
            # capture_height=720,
            # framerate=60,
            flip_method=0,
            display_width=960,
            display_height=540,
        )
    )
    left_camera.K = np.array([[779.86083781,   0.        , 445.90364404],
                             [0.         ,  779.65153775, 269.26453119],
                             [0.         ,  0.          , 1.          ]])
    left_camera.dist_coeff = np.array([-0.329970284,  0.0813242658, -0.000012252, 0.00293898426, 0.102042450])
    left_camera.start()

    right_camera = CSI_Camera()
    right_camera.open(
        gstreamer_pipeline(
            sensor_id=0,
            # capture_width=1280,
            # capture_height=720,
            # framerate=60,
            flip_method=0,
            display_width=960,
            display_height=540,
        )
    )
    right_camera.K = np.array([[814.9685752,   0.        , 487.09541315],
                             [0.         ,  813.76140548, 252.14976917],
                             [0.         ,  0.          , 1.          ]])
    right_camera.dist_coeff = np.array([-0.329970284,  0.0813242658, -0.000012252, 0.00293898426, 0.102042450])
    right_camera.start()
    

    if left_camera.video_capture.isOpened() and right_camera.video_capture.isOpened():

        cv.namedWindow(window_title, cv.WINDOW_AUTOSIZE)

        try:
            count = 0

            while True:
                # get origin camera data
                _, left_image_orin = left_camera.read()
                _, right_image_orin = right_camera.read()

                # left_image_orin= np.dstack((cv.equalizeHist(left_image_orin[:,:,0]),
                #                              cv.equalizeHist(left_image_orin[:,:,1]),
                #                              cv.equalizeHist(left_image_orin[:,:,2])))
                
                # right_image_orin= np.dstack((cv.equalizeHist(right_image_orin[:,:,0]),
                #                              cv.equalizeHist(right_image_orin[:,:,1]),
                #                              cv.equalizeHist(right_image_orin[:,:,2])))

                # img_cvt_l = cv.cvtColor(left_image_orin, cv.COLOR_BGR2YCrCb)
                # img_cvt_r = cv.cvtColor(right_image_orin, cv.COLOR_BGR2YCrCb)

                # img_hist_l = np.dstack((cv.equalizeHist(img_cvt_l[:,:,0]),
                #                              cv.equalizeHist(img_cvt_l[:,:,1]),
                #                              cv.equalizeHist(img_cvt_l[:,:,2])))
                
                # img_hist_r = np.dstack((cv.equalizeHist(img_cvt_r[:,:,0]),
                #                              cv.equalizeHist(img_cvt_r[:,:,1]),
                #                              cv.equalizeHist(img_cvt_r[:,:,2])))
                
                # left_image_orin = cv.cvtColor(img_hist_l, cv.COLOR_YCrCb2BGR)
                # right_image_orin = cv.cvtColor(img_hist_r, cv.COLOR_YCrCb2BGR)

                # define camera distortion data
                
                # image reshaping and undistorting
                h_l, w_l = left_image_orin.shape[:2]
                h_r, w_r = right_image_orin.shape[:2]

                new_camera_matrix_l, roi_l = cv.getOptimalNewCameraMatrix(left_camera.K, left_camera.dist_coeff, (w_l,h_l), 1, (w_l,h_l))
                new_camera_matrix_r, roi_r = cv.getOptimalNewCameraMatrix(right_camera.K, right_camera.dist_coeff, (w_r,h_r), 1, (w_r,h_r))
                left_image = cv.undistort(left_image_orin, left_camera.K, left_camera.dist_coeff, None, new_camera_matrix_l)
                right_image = cv.undistort(right_image_orin, right_camera.K, right_camera.dist_coeff, None, new_camera_matrix_r)

                x,y,w_l,h_l = roi_l
                left_image = left_image[y:y+h_l, x:x+w_l]
                x,y,w_r,h_r = roi_r
                right_image = right_image[y:y+h_r, x:x+w_r]

                if(count==0):
                    # get feature points to realtime camera image stitching
                    brisk = cv.BRISK_create()

                    # keypoints1, descriptors1 = 0
                    # keypoints2, descriptors2 = 0


                    keypoints1, descriptors1 = brisk.detectAndCompute(left_image, None)
                    keypoints2, descriptors2 = brisk.detectAndCompute(right_image, None)
                    count+=1              

                    fmatcher = cv.DescriptorMatcher_create('BruteForce-Hamming')
                    match = fmatcher.match(descriptors1, descriptors2)

                    pts1, pts2 = [], []
                    window_size = (left_image.shape[1]*2-218, left_image.shape[0])    # window_size = 1600x475
        
                    for i in range(len(match)):
                        pts1.append(keypoints1[match[i].queryIdx].pt)
                        pts2.append(keypoints2[match[i].trainIdx].pt)
                    pts1 = np.array(pts1, dtype=np.float32)
                    pts2 = np.array(pts2, dtype=np.float32)

                    H, inlier_mask = cv.findHomography(pts2, pts1, cv.RANSAC)
                    # print(H)
                img_merged = cv.warpPerspective(right_image, H, window_size)
                    # print(window_size)
                img_merged[:,:left_image.shape[1]] = left_image    #Copy

                # camera_images = np.hstack((left_image, right_image)) 
                # Check to see if the user closed the window
                # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
                # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
                if cv.getWindowProperty(window_title, cv.WND_PROP_AUTOSIZE) >= 0:
                    cv.imshow(window_title, img_merged)
                    # cv.imshow(window_title, camera_images)
                else:
                    break

                # This also acts as
                keyCode = cv.waitKey(30) & 0xFF
                # Stop the program on the ESC key
                if keyCode == 27:
                    break
        finally:

            left_camera.stop()
            right_camera.stop()
            left_camera.release()
            right_camera.release()
        cv.destroyAllWindows()
    else:
        print("Error: Unable to open both cameras")
        left_camera.stop()
        right_camera.stop()
        left_camera.release()
        right_camera.release()



if __name__ == "__main__":
    run_cameras()

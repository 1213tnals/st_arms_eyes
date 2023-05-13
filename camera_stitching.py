import cv2 as cv
import numpy as np

""" 
gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
Flip the image by setting the flip_method (most common values: 0 and 2)
display_width and display_height determine the size of each camera pane in the window on the screen
Default 1920x1080 displayd in a 1/4 size window
"""

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
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


def show_stitched_camera():
    window_title = "CSI Camera"

    # print(gstreamer_pipeline(flip_method=0))
    left_cam = cv.VideoCapture(gstreamer_pipeline(sensor_id=0, flip_method=0), cv.CAP_GSTREAMER)
    right_cam = cv.VideoCapture(gstreamer_pipeline(sensor_id=1, flip_method=0), cv.CAP_GSTREAMER)

    # fourcc = cv.VideoWriter_fourcc(*'mp4v')
    # width_l = left_cam.get(cv.CAP_PROP_FRAME_WIDTH)
    # height_l = left_cam.get(cv.CAP_PROP_FRAME_HEIGHT)
    # width_r = right_cam.get(cv.CAP_PROP_FRAME_WIDTH)
    # height_r = right_cam.get(cv.CAP_PROP_FRAME_HEIGHT)
    # size_l = (int(width_l), int(height_l))
    # size_r = (int(width_r), int(height_r))

    # out = cv.VideoWriter('output_l.mp4', fourcc, 20.0, size_l)
    # out = cv.VideoWriter('output_R.mp4', fourcc, 20.0, size_r)

    while True:
        ret_val_l, left_img = left_cam.read()
        ret_val_r, right_img = right_cam.read()
                # Check to see if the user closed the window
                # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
                # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
    
        brisk = cv.BRISK_create()
        print("eeee")
        keypoints1, descriptors1 = brisk.detectAndCompute(left_img, None)
        print("iiii")
        keypoints2, descriptors2 = brisk.detectAndCompute(right_img, None)

        fmatcher = cv.DescriptorMatcher_create('BruteForce-Hamming')
        match = fmatcher.match(descriptors1, descriptors2)


        # Calculate planar homography and merge them
        pts1, pts2 = [], []
        
        for i in range(len(match)):
            pts1.append(keypoints1[match[i].queryIdx].pt)
            pts1.append(keypoints2[match[i].queryIdx].pt)
        pts1 = np.array(pts1, dtype=np.float32)
        pts2 = np.array(pts2, dtype=np.float32)

        H, inlier_mask = cv.finHomography(pts2, pts1, cv.RANSAC)
        img_merged = cv.warpPerspective(right_img, H, (left_img.shape[1]*2, right_img.shape[0]))
        img_merged[:,:left_img.shape[1]] = left_img    #Copy



        if ret_val_l and ret_val_r:
            cv.imshow(window_title, img_merged)
            # out.write(frame)

            keyCode = cv.waitKey(10) & 0xFF
            # Stop the program on the ESC key or 'q'
            if keyCode == 27 or keyCode == ord('q'):
                break

    left_cam.release()
    right_cam.release()
    # out.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    show_stitched_camera()
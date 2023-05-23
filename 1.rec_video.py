# MIT License
# Copyright (c) 2019-2022 JetsonHacks

# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image

import cv2 as cv

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


def show_camera():
    window_title = "CSI Camera"

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    # option1. record left camera1
    # video_capture = cv.VideoCapture(gstreamer_pipeline(sensor_id=1, flip_method=0), cv.CAP_GSTREAMER)
    # option2. record right camera1
    video_capture = cv.VideoCapture(gstreamer_pipeline(sensor_id=0, flip_method=0), cv.CAP_GSTREAMER)

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    width = video_capture.get(cv.CAP_PROP_FRAME_WIDTH)
    height = video_capture.get(cv.CAP_PROP_FRAME_HEIGHT)
    size = (int(width), int(height))

    # option1. record left camera2
    # out = cv.VideoWriter('output_l.mp4', fourcc, 20.0, size)
    # option2. record right camera2
    out = cv.VideoWriter('data/output_r.mp4', fourcc, 20.0, size)

    while True:
        ret_val, frame = video_capture.read()
                # Check to see if the user closed the window
                # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
                # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
        if ret_val:
            cv.imshow(window_title, frame)
            out.write(frame)

            keyCode = cv.waitKey(10) & 0xFF
            # Stop the program on the ESC key or 'q'
            if keyCode == 27 or keyCode == ord('q'):
                break

    video_capture.release()
    out.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    show_camera()
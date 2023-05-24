## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2 as cv

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# if device_product_line == 'L500':
#     config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
# else:
#     config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        depth_center_x, depth_center_y = (0,0)
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())    # depth_image = height: 480, width: 640 shape x 1, and data unit is mm data, dtype=uint16
        color_image = np.asanyarray(color_frame.get_data())    # color_image = height: 480, width: 640 shape x 3, and data is color
        # print(depth_image.dtype)
        depth_image[depth_image > 300] = 0                     # tresholding
        depth_image[depth_image != 0] = 1                      # binary
        depth_image_8bit = depth_image.astype(np.uint8)
        
        contours, hierarchy = cv.findContours(depth_image_8bit, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # if len(contours) > 0:
        #     contour = contours[0]
        #     M = cv.moments(contour)
        #     if(M["m00"]==0):
        #         M["m00"]=1
        #     cX = int(M["m10"]/ M["m00"])
        #     cY = int(M["m01"]/ M["m00"])
        #     depth_center_x, depth_center_y = cX, cY

        if(np.sum(depth_image_8bit)!=0):
            _, labels, stats, centroids = cv.connectedComponentsWithStats(depth_image_8bit)
            depth_center_x, depth_center_y = centroids[1]
        print("center: ({:.3f},{:.3f})".format(depth_center_x, depth_center_y))
        
        for i in range(color_image.shape[2]):
            color_image[:,:,i] = color_image[:,:,i]*depth_image_8bit
        # print(depth_image_8bit[240][320])
        # depth_image_3 = np.expand_dims(depth_image, axis=2)
        # depth_image_3D = np.concatenate((depth_image_3, depth_image_3), axis=2)
        # print(depth_image_3D.shape)
        # color_image_treshold = cv.merge([cv.threshold(color_image[:,:,0], depth_image_8bit, 255, cv.THRESH_BINARY)[1],
        #                                  cv.threshold(color_image[:,:,1], depth_image_8bit, 255, cv.THRESH_BINARY)[1],
        #                                  cv.threshold(color_image[:,:,2], depth_image_8bit, 255, cv.THRESH_BINARY)[1]])
        # print(depth_image[240][320])                         # center depth data

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        cv.namedWindow('st_arm_eyes', cv.WINDOW_AUTOSIZE)
        cv.circle(color_image, (np.uint8(depth_center_x), np.uint8(depth_center_y)), 10, (0,0,255), 3)
        contour_image = cv.drawContours(color_image, contours, -1, (255, 0, 0), 2)
        print(contours)
        cv.imshow('st_arm_eyes', contour_image)
        
        # This also acts as
        keyCode = cv.waitKey(30) & 0xFF
        # Stop the program on the ESC key
        if keyCode == 27:
            break

finally:

    # Stop streaming
    pipeline.stop()
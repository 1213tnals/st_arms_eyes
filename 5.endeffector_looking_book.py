## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2 as cv

cover_file = 'book.jpg'
f, cx, cy = 1000, 320, 240
min_inlier_num = 100

# Load an image
img_file = 'book.jpg'
obj_img = cv.imread(img_file)      # book is 3D object image
assert obj_img is not None, 'Cannot read the given image, ' + img_file

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
pipeline.start(config)

fdetector = cv.ORB_create()
fmatcher = cv.DescriptorMatcher_create('BruteForce-Hamming')

# Load the object image and extract features
obj_image = cv.imread(cover_file)
assert obj_image is not None, 'Cannot read the given cover image, ' + cover_file
obj_keypoints, obj_descriptors = fdetector.detectAndCompute(obj_image, None)
assert len(obj_keypoints) >= min_inlier_num, 'The given cover image contains too small number of features.'
fmatcher.add(obj_descriptors)

# Prepare a box for simple AR
box_lower = np.array([[180, 90, 0], [180, 350, 0], [250, 350, 0], [250, 90, 0]], dtype=np.float32)
box_upper = np.array([[180, 90, -50], [180, 350, -50], [250, 350, -50], [250, 90, -50]], dtype=np.float32)

# Run pose extimation
K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
dist_coeff = np.zeros(5)

while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())    # depth_image = height: 480, width: 640 shape, and data is mm data
    color_image = np.asanyarray(color_frame.get_data())
    print(depth_image[240][320])                         # center depth data

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)
    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = color_image.shape


    # Extract features and match them to the object features
    cam_keypoints, cam_descriptors = fdetector.detectAndCompute(color_image, None)
    match = fmatcher.match(cam_descriptors, obj_descriptors)
    if len(match) < min_inlier_num:
        continue

    obj_pts, cam_pts = [], []
    for m in match:
        obj_pts.append(obj_keypoints[m.trainIdx].pt)
        cam_pts.append(cam_keypoints[m.queryIdx].pt)
    obj_pts = np.array(obj_pts, dtype=np.float32)
    obj_pts = np.hstack((obj_pts, np.zeros((len(obj_pts), 1), dtype=np.float32)))    # Make 2D to 3D
    cam_pts = np.array(cam_pts, dtype=np.float32)

    # Deterimine whether each matched feature is an inlier or not
    ret, rvec, tvec, inliers = cv.solvePnPRansac(obj_pts, cam_pts, K, dist_coeff, useExtrinsicGuess=False,
                                                 iterationsCount=500, reprojectionError=2., confidence=0.99)
    inlier_mask = np.zeros(len(match), dtype=np.uint8)
    inlier_mask[inliers] = 1
    ## img_result = cv.drawMatches(color_image, cam_keypoints, obj_image, obj_keypoints, match, None, (0, 0, 0), (0, 0, 0), inlier_mask)

    # Check whether inliers are enough or not
    inlier_num = sum(inlier_mask)
    # Box pose visible when the camera center distance are in under 500mm
    if inlier_num > min_inlier_num and depth_image[240-60][320-80] < 500 and depth_image[240+60][320+80]< 500:
        # Estimate camera pose with inliers
        ret, rvec, tvec = cv.solvePnP(obj_pts[inliers], cam_pts[inliers], K, dist_coeff)

        # Draw the box on the image
        line_lower, _ = cv.projectPoints(box_lower, rvec, tvec, K, dist_coeff)
        line_upper, _ = cv.projectPoints(box_upper, rvec, tvec, K, dist_coeff)
        cv.polylines(color_image, [np.int32(line_lower)], True, (255, 0, 0), 2)
        cv.polylines(color_image, [np.int32(line_upper)], True, (0, 0, 255), 2)
        for b, t in zip(line_lower, line_upper):
            cv.line(color_image, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2)

    # Show the image and process the key event
    info = f'rvec: {rvec}, tvec: {tvec}'         # rvec unit is rad
    cv.putText(color_image, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
    cv.imshow('Pose Estimation (Book)', color_image)
    

    keyCode = cv.waitKey(30) & 0xFF
        # Stop the program on the ESC key
    if keyCode == 27:
        break


# Stop streaming
pipeline.stop()
cv.destroyAllWindows()
## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import math

def initializeCamera():
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

    # I don't know why there are bug if I use camera with IMU. So I didn't use both sensor same time. 
    # config.enable_stream(rs.stream.accel)
    # config.enable_stream(rs.stream.gyro)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    prof = pipeline.start(config)
    
    return pipeline

def gyro_data(gyro):
    return np.asarray([gyro.x, gyro.y, gyro.z])

def accel_data(accel):
    return np.asarray([accel.x, accel.y, accel.z])

def change_axis_t(translation_tvec):
    return np.array([translation_tvec[0],translation_tvec[1],-translation_tvec[2]])

def change_axis_r(rotation_rvec):
    return np.array([rotation_rvec[0],rotation_rvec[1],-rotation_rvec[2]])

def rad2deg(rotation_rvec_rad):
    return np.array([math.degrees(rotation_rvec_rad[0]),math.degrees(rotation_rvec_rad[1]),math.degrees(rotation_rvec_rad[2])])
# cover_file = 'data/book.jpg'





# fdetector = cv.ORB_create()
# fmatcher = cv.DescriptorMatcher_create('BruteForce-Hamming')

# # Load the object image and extract features
# obj_image = cv.imread(cover_file)
# assert obj_image is not None, 'Cannot read the given cover image, ' + cover_file
# obj_keypoints, obj_descriptors = fdetector.detectAndCompute(obj_image, None)
# assert len(obj_keypoints) >= min_inlier_num, 'The given cover image contains too small number of features.'
# fmatcher.add(obj_descriptors)

# # Prepare a box for simple AR
# box_lower = np.array([[180, 90, 0], [180, 350, 0], [250, 350, 0], [250, 90, 0]], dtype=np.float32)
# box_upper = np.array([[180, 90, -50], [180, 350, -50], [250, 350, -50], [250, 90, -50]], dtype=np.float32)

# # Run pose extimation



# min_inlier_num = 100
# Set camera and checker board information

f, cx, cy = 1000, 320, 240
K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
dist_coeff = np.zeros(5)
board_pattern = (8, 6)
board_cellsize = 0.025
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# Prepare a 3D box for simple AR
box_lower = board_cellsize * np.array([[1, 1,  0], [6, 1,  0], [6, 4,  0], [1, 4,  0]])
box_upper = board_cellsize * np.array([[1, 1, -1], [6, 1, -1], [6, 4, -1], [1, 4, -1]])
magic_field1 = board_cellsize * np.array([[4, 0, -3], [2, 4, -3], [6, 4, -3]])
magic_field2 = board_cellsize * np.array([[4, 5, -3], [2, 1, -3], [6, 1, -3]])

# Prepare 3D points on a chessboard
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])  # 8x6 행렬 생성

# Set Refference Pose
ref_rvec = np.array([[0],[0],[0]])
ref_tvec = np.array([[-0.09], [-0.07], [-0.5]])


pipeline = initializeCamera()
cnt = 1

try:
    while True:
        frames = pipeline.wait_for_frames()

        # Get IMU data
        # accel = accel_data(frames[0].as_motion_frame().get_motion_data())
        # gyro = gyro_data(frames[1].as_motion_frame().get_motion_data())

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())    # depth_image = height: 480, width: 640 shape, and data is mm data
        color_image = np.asanyarray(color_frame.get_data())
        # print(depth_image[240][320])                           # center depth data

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv.convertScaleAbs(depth_image, alpha=0.03)
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        complete, board_points = cv.findChessboardCorners(color_image, board_pattern, board_criteria)
        if complete:
            ret, rvec, tvec = cv.solvePnP(obj_points, board_points, K, dist_coeff)        # obj: 보드의 3D 좌표 / board_point: 체스보드나 물체의 2D 좌표, 이후 왜곡 고려
            aligned_tvec = change_axis_t(tvec)
            aligned_rvec = change_axis_r(rvec)
            translation_error = ref_tvec - aligned_tvec
            rotation_error = ref_rvec - aligned_rvec
            rotation_error_deg = rad2deg(rotation_error)

            info_rpy = f"rpy_error: [{rotation_error_deg[0]:.4f} {rotation_error_deg[1]:.4f} {rotation_error_deg[2]:.4f}]"
            info_xyz = f"xyz_error: [{translation_error[0]} {translation_error[1]} {translation_error[2]}]"
            cv.putText(color_image, info_rpy, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), thickness=2)
            cv.putText(color_image, info_rpy, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 0), thickness=1)
            cv.putText(color_image, info_xyz, (10, 50), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), thickness=2)
            cv.putText(color_image, info_xyz, (10, 50), cv.FONT_HERSHEY_DUPLEX, 0.6, (255, 0, 255), thickness=1)

            if(abs(rotation_error[2])>10 or abs(translation_error[0])>-0.01/aligned_tvec[2] or abs(translation_error[1])>-0.01/aligned_rvec[2]):
                info = 'first: x,y,yaw error to be completed'
            else:
                info = 'second: z,roll,pitch error to be completed'
            print(-0.01/aligned_tvec[2])
            
            cv.putText(color_image, info, (10, 455), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), thickness=2)
            cv.putText(color_image, info, (10, 455), cv.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), thickness=1)
            # If you want to know camera pose by image(checker board) use it
            # R, _ = cv.Rodrigues(rvec) # Alternative) scipy.spatial.transform.Rotation
            # p = (-R.T @ tvec).flatten()
            
        # Show the image and process the key event
        cv.imshow('Pose Estimation (Checkerboard)', color_image)
        key = cv.waitKey(10)
        if key == ord(' '):
            key = cv.waitKey()
        if key == 27: # ESC
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv.destroyAllWindows()
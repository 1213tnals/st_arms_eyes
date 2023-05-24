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
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())    # depth_image = height: 480, width: 640 shape, and data unit is mm data
        color_image = np.asanyarray(color_frame.get_data())
        depth_image[depth_image > 300] = 0                     # tresholding
        # depth_image[depth_image != 0] = 1                      # binary
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # Show images
        cv.namedWindow('st_arm_eyes', cv.WINDOW_AUTOSIZE)
        cv.imshow('st_arm_eyes', depth_colormap)
        
        # This also acts as
        keyCode = cv.waitKey(30) & 0xFF
        # Stop the program on the ESC key
        if keyCode == 13:
            cv.imwrite('depth3.png', depth_colormap)
            print("save!")
        if keyCode == 27:
            break

finally:

    # Stop streaming
    pipeline.stop()
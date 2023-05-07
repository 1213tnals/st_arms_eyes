import numpy as np
import cv2 as cv

# The given video and calibration data
input_file = 'output.mp4'
# photo_file = 'data/1.jpg'
# photo = cv.imread(photo_file)
# height, width, _ = photo.shape
# photo_size=(width/100,height/100)

K = np.array([[859.81674109,   0.        , 494.44231245],
 [0.         ,  855.69850833, 221.30750395],
 [0.         ,  0.          , 1.          ]])
dist_coeff = np.array([-0.37838205,  -0.68771675, 0.00982567, -0.00609295, 3.55980975])
board_pattern = (8, 6)
board_cellsize = 0.025
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# Open a video
video = cv.VideoCapture(input_file)
assert video.isOpened(), 'Cannot read the given input, ' + input_file

# Prepare a 3D box for simple AR
box_lower = board_cellsize * np.array([[1, 1,  0], [6, 1,  0], [6, 4,  0], [1, 4,  0]])
box_upper = board_cellsize * np.array([[1, 1, -1], [6, 1, -1], [6, 4, -1], [1, 4, -1]])
magic_field1 = board_cellsize * np.array([[4, 0, -3], [2, 4, -3], [6, 4, -3]])
magic_field2 = board_cellsize * np.array([[4, 5, -3], [2, 1, -3], [6, 1, -3]])

# Prepare 3D points on a chessboard
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])  # 8x6 행렬 생성
# photo_dst = np.array([[0, 0], [photo_size[0], 0], [0, photo_size[1]], [photo_size[0], photo_size[1]]])
# photo_points = board_cellsize * np.array([[1, 1, -1], [1+photo_size[0], 1, -1], [1+photo_size[0], 1+photo_size[1], -1], [1+0, 1+photo_size[1], -1],])

# Run pose estimation
while True:
    # Read an image from the video
    valid, board = video.read()
    if not valid:
        break

    # Estimate the camera pose
    complete, board_points = cv.findChessboardCorners(board, board_pattern, board_criteria)
    if complete:
        ret, rvec, tvec = cv.solvePnP(obj_points, board_points, K, dist_coeff)        # obj: 보드의 3D 좌표 / board_point: 체스보드나 물체의 2D 좌표, 이후 왜곡 고려
        # ret, rvec, tvec = cv.solvePnP(photo_points, board_points, K, dist_coeff)    # 체커보드와 이미지 사이의 R, T 행렬을 구함

        # Draw the box on the image
        magic_line1, _ = cv.projectPoints(magic_field1, rvec, tvec, K, dist_coeff)
        magic_line2, _ = cv.projectPoints(magic_field2, rvec, tvec, K, dist_coeff)
        cv.polylines(board, [np.int32(magic_line1)], True, (0,255,0), 4)
        cv.polylines(board, [np.int32(magic_line2)], True, (255,0,127), 4)
        # photo_upper, _ = cv.projectPoints(photo_points, rvec, tvec, K, dist_coeff)

        
        # H, _ = cv.findHomography(photo_dst,photo_points)
        # print(H)
        # img_rectify = cv.warpPerspective(photo, H, (100,100))
        
        # H, _ = cv.findHomography(photo_points, photo_upper)
        # img_rectify = cv.warpPerspective(photo, H, photo_size)
        # for b, t in zip(magic_line1, magic_line2):
        #     cv.line(board, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2)

        # Print the camera position
        R, _ = cv.Rodrigues(rvec) # Alternative) scipy.spatial.transform.Rotation
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(board, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    # Show the image and process the key event
    cv.imshow('Pose Estimation (Checkerboard)', board)
    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27: # ESC
        break

video.release()
cv.destroyAllWindows()
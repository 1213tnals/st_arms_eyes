import cv2 as cv
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

depth_image1 = cv.imread('depth1.png', cv.IMREAD_GRAYSCALE)
depth_image2 = cv.imread('depth2.png', cv.IMREAD_GRAYSCALE)
depth_image3 = cv.imread('depth3.png', cv.IMREAD_GRAYSCALE)

def detect_keypoints(image):
    keypoints = cv.cornerHarris(image, 2, 3, 0.04)
    keypoints = cv.dilate(keypoints, None)
    keypoints = keypoints > 0.01 * keypoints.max()
    return keypoints

keypoints1 = detect_keypoints(depth_image1)
keypoints2 = detect_keypoints(depth_image2)
keypoints3 = detect_keypoints(depth_image3)

points1 = np.argwhere(keypoints1)
points2 = np.argwhere(keypoints2)
points3 = np.argwhere(keypoints3)

point_cloud = np.vstack((points1, points2, points3))
tri = Delaunay(point_cloud)

print(points1[0,0])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], c='b', marker='o')
ax.plot_trisurf(point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], triangles=tri.simplices, color='r')

ax.set_xlabel('X')
ax.set_xlabel('Y')
ax.set_xlabel('Z')

plt.show()
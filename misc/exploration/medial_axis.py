import numpy as np
from skimage import morphology, draw, measure
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import networkx as nx

def medial_axis(img):
    # invert the image to treat the free space as foreground
    img = img.astype(bool)
    # compute the medial axis
    skel, distance = morphology.medial_axis(img, return_distance=True)
    # create a blank grayscale image to store the medial axis
    med_axis = np.zeros(img.shape, dtype=np.uint8)
    # set the pixel values of the medial axis based on the distance transform
    med_axis[skel] = 255
    # return the medial axis image
    return med_axis

def find_medial_axis_vertices(med_axis):
    # find the contours of the medial axis image
    contours = measure.find_contours(med_axis, 0.5)
    # find the center of each contour
    centers = [np.mean(contour, axis=0) for contour in contours]
    vertices = centers
    # vertices.extend(intersections)
    '''
    vertices = []
    # approximate each contour with a polygon
    for contour in contours:
        polygon = measure.approximate_polygon(contour, tolerance=2.0)
        vertices.extend(polygon.tolist())
    # remove duplicate vertices
    vertices = list(set(tuple(v) for v in vertices))
    # return the vertices
    '''
    # merge vertices that are close to each other
    dist_matrix = cdist(vertices, vertices)
    for i, v1 in enumerate(vertices):
        for j, v2 in enumerate(vertices[i+1:], start=i+1):
            if dist_matrix[i, j] < 3:
                vertices[i] = v1
                vertices[j] = v1
    return vertices, contours




def is_edge_intersecting_obstacle(v1, v2, occupancy_grid):
    # get the line segment defined by the two vertices
    x1, y1 = v1
    x2, y2 = v2
    x_min = int(min(x1, x2))
    x_max = int(max(x1, x2))
    y_min = int(min(y1, y2))
    y_max = int(max(y1, y2))
    # check each pixel along the line segment
    for x in range(x_min, x_max+1):
        for y in range(y_min, y_max+1):
            if occupancy_grid[x, y] == 0:
                # the pixel is occupied by an obstacle, so the edge intersects with an obstacle
                return True
    # the edge does not intersect with any obstacles
    return False

img = cv2.imread("occupancy_grid_framed.jpg")
img_copy = img.copy()
# make grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img//240
# compute the medial axis of the free space
med_axis = medial_axis(img)
'''
vertices = []
for x in range(med_axis.shape[0]):
    for y in range(med_axis.shape[1]):
        if med_axis[x, y] == 255:
            neighbors = med_axis[max(0, x-2):min(med_axis.shape[0], x+2),
                                    max(0, y-2):min(med_axis.shape[1], y+2)]
            print(neighbors)
            if np.sum(neighbors == 255) > 4:
                vertices.append((x, y))
                '''
vertices,cont = find_medial_axis_vertices(med_axis)
# plan a path through the medial axis
#path = plan_path(vertices, img)

# display the results
fig, axes = plt.subplots(1, 2, figsize=(8, 4)) 
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Input')
axes[1].imshow(med_axis, cmap='gray')
axes[0].plot([v[1] for v in vertices], [v[0] for v in vertices], 'r.')
axes[1].set_title('Medial Axis')
plt.show()
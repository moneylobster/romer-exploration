from shapely.geometry import Polygon, LineString
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import cv2

# Load the image
img = cv2.imread("occupancy_grid_framed.jpg")
# make a copy of the image
img_copy = img.copy()
#make it grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape)
# img = np.rot90(img)
# flip the image
# img = np.flipud(img)

# Find the coordinates of the black pixels
coords = np.column_stack(np.where(img == 0))

edges = cv2.Canny(img, 50, 150)

# Find the contours of the obstacles
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

coords2= np.empty((0,2), int)
for contour in contours:
    for point in contour:
        coords2 = np.append(coords2, point, axis=0)
coords2 = coords2[1:,:]
coords2 = np.unique(coords2, axis=0)
#
print(np.max(coords2, axis=0))

# Display the original image with the obstacle boundaries marked in green
# cv2.imwrite("occupancy_grid_contours.jpg", img_with_contours)



# Compute the Voronoi diagram
vor = Voronoi(coords2)


# Plot the Voronoi diagram on top of the image
fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6, point_size=2)
plt.imshow(img_copy, zorder=0, extent=[0, 45, 16, 0])
plt.show()

import numpy as np
from skimage import io
from skimage import filter
from skimage import morphology
from skimage import segmentation
from skimage import measure
from skimage import draw
import matplotlib.pyplot as plt

img = io.imread('../coins_black_small.jpg', as_grey=True)
val = filter.threshold_otsu(img)

mask = img < val

# Compute compacity of granular material
compacity = mask[100:-100, 100:700].mean()
print "compacity is ", compacity

# Separate the different coins
erosion = morphology.binary_erosion(mask, morphology.disk(9))
erosion = morphology.binary_erosion(erosion, morphology.disk(5))


labs = morphology.label(erosion, background=0)
labs += 1

edges = filter.sobel(img)

regions = morphology.watershed(edges, markers=labs, mask=mask)

# remove borders
l0, l1 = img.shape
indices_borders = [regions[0, 0], regions[0, l1/2], \
                            regions[l0 - 1, l1/2]]

for index in indices_borders:
    regions[regions == index] = 0

seg, _, _ = segmentation.relabel_from_one(regions)

# Compute neighbors

neighbors = []

for lab in range(1, seg.max() + 1):
    dilation_of_region = morphology.binary_dilation(seg == lab,
                                morphology.diamond(3))
    neighbors.append(np.unique(seg[dilation_of_region]))

res = measure.regionprops(seg, ['Area', 'Centroid'])
areas = np.array([entry['Area'] for entry in res])
centroids = np.array([entry['Centroid'] for entry in res])

number_of_neighbors = np.array([len(el) - 2 for el in neighbors])

new = np.zeros_like(img)


for lab_index, neighbors_index in enumerate(neighbors):
    true_neighbors = np.setdiff1d(neighbors_index, [0, lab_index + 1])
    x, y = centroids[lab_index]
    for element in true_neighbors:
        x1, y1 = centroids[element - 1]
        inds = draw.line(int(x), int(y), int(x1), int(y1))
        new[inds] = 1

plt.imshow(img, cmap='gray', interpolation='nearest')
plt.contour(new, [0.5], linewidths=4)
plt.show()

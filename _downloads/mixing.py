import numpy as np
from skimage import io
from skimage import filter
from skimage import morphology
from glob import glob
import matplotlib.pyplot as plt

def mixing_region(filename):
    white = io.imread(filename)
    val = filter.threshold_otsu(white)
    light_mask = white > val
    regions = morphology.label(light_mask)
    index_large_region = np.argmax(np.bincount(regions.ravel()))
    fluid_mask = regions == index_large_region
    fluid_mask = morphology.binary_erosion(fluid_mask, selem=np.ones((3, 3)))
    return fluid_mask


mixing_list = glob('../mixing_images/*.JPG')
mixing_list.sort()

white_list = glob('../white_images/*.JPG')
white_list.sort()

# Compute fluid region
# ---------------------------------

white = io.imread(white_list[0])
val = filter.threshold_otsu(white)
light_mask = white > val
regions = morphology.label(light_mask)
index_large_region = np.argmax(np.bincount(regions.ravel()))
fluid_mask = regions == index_large_region
fluid_mask = morphology.binary_erosion(fluid_mask, selem=np.ones((5, 5)))

# Compute concentration field
# -----------------------------------

#for filename in mixing_list:
#    img = io.iomread(filename)
#    conc = np.log(1)

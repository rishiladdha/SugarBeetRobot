
#%% imports
import os
import numpy as np
from skimage import io as skio
from skimage import filters, measure, morphology
from sklearn import cluster
import matplotlib.pyplot as plt
import scipy

# %%
basedir = 'bonirob_2016-05-23-10-47-22_2_im'
rgb_dir = os.path.join(basedir, 'rgb')
nir_dir = os.path.join(basedir, 'nir')

rgb_files = sorted(os.listdir(rgb_dir))
nir_files = sorted(os.listdir(nir_dir))

rgb_list = [skio.imread(os.path.join(rgb_dir, f)) for f in rgb_files]
nir_list = [skio.imread(os.path.join(nir_dir, f)) for f in nir_files]
print(rgb_list[0])

# %% Print shapes of the image arrays
print('Shapes: RGB:', rgb_list[5].shape, 'NIR:', nir_list[5].shape)

#%% Display RGB image
fig, ax = plt.subplots()
ax.imshow(rgb_list[5])
fig.suptitle('RGB image', fontdict={'color': 'blue'})
fname = 'rgb_image.png'
fig.savefig(fname)
fname

#%% Display NIR image by mapping the values to shades of gray
fig, ax = plt.subplots()
ax.imshow(nir_list[5], cmap='gray')
fig.suptitle('NIR image', fontdict={'color': 'blue'})
fname = 'nir_image.png'
fig.savefig(fname)
fname
#why cmap='gray'?

#%% Display the channels
fig, axes = plt.subplots(nrows=3, ncols=2, sharex='all', sharey='all')
for jj, color in enumerate(('Red', 'Green', 'Blue')):
    axes[jj, 0].imshow(rgb_list[5][:, :, jj],cmap='gray')
    axes[jj, 0].set_title(color)
axes[0, 1].imshow(rgb_list[5])
axes[0, 1].set_title('RGB')
axes[1, 1].imshow(nir_list[5], cmap='gray')
axes[1, 1].set_title('NIR')
axes[2, 1].remove()
fname = 'all_channels.png'
fig.set_size_inches(6, 8)
for ax in axes.flat:
    ax.set_axis_off()
fig.tight_layout()
fig.savefig(fname)
fname

# %%Histogram for manual threshold detection
fig, ax = plt.subplots()
ax.hist(nir_list[5].ravel(), bins=256)
ax.set_title('Histogram of NIR image')
ax.set_xlabel('Pixel value')
ax.set_ylabel('Number of pixels')
fname = 'histogram.png'
fig.savefig(fname)
fname

# %%
thresh = filters.threshold_otsu(nir_list[5])
print(thresh)

# %% thresholding the image
binary = nir_list[5] > thresh
fig, ax = plt.subplots()
ax.imshow(binary, cmap='gray')
ax.set_title('Binary mask')
fname = 'binary_mask.png'
fig.savefig(fname)
fname

# %% morphology computation
morphed = morphology.remove_small_objects(binary, 100)
#erosion
morphed = morphology.binary_erosion(morphed, morphology.disk(2))
#dilation
morphed = morphology.binary_dilation(morphed, morphology.disk(2))
#opening
morphed = morphology.binary_opening(morphed, footprint=morphology.diamond(5))
#closing
morphed = morphology.binary_closing(morphed, morphology.disk(2))

fig, ax = plt.subplots()
ax.imshow(morphed, cmap='gray')
ax.set_title('Opening of the binary image')
fname = 'morphed_mask.png'
fig.savefig(fname)
fname

#%% Create the label image
label_im = measure.label(morphed, background=0)
fig, axes = plt.subplots(nrows=1, ncols=2, sharex='all', sharey='all')
axes[0].set_title('Color-mapped Labels')
axes[0].imshow(label_im, cmap='jet')
axes[1].imshow(nir_list[5], cmap='gray')
axes[1].set_title('Original NIR image')
fname = 'labeled_image.png'
fig.set_size_inches(8, 4)
fig.savefig(fname)
fname

#%% Extract and display region properties
fig, ax = plt.subplots()
ax.imshow(label_im)
props = measure.regionprops(label_im)
for prop in props:
    y0, x0, y1, x1 = prop.bbox
    plt.plot((x0, x1, x1, x0, x0), (y0, y0, y1, y1, y0), 'k-')
    y0, x0 = prop.centroid
    x1 = x0 + np.cos(prop.orientation) * 0.5 * prop.minor_axis_length   
    y1 = y0 - np.sin(prop.orientation) * 0.5 * prop.minor_axis_length
    x2 = x0 - np.sin(prop.orientation) * 0.5 * prop.major_axis_length
    y2 = y0 - np.cos(prop.orientation) * 0.5 * prop.major_axis_length
    ax.plot((x0, x1), (y0, y1), '-m')
    ax.plot((x0, x2), (y0, y2), '-r')
fname = 'labeled_image_geom.png'
fig.savefig(fname)
fname

#%% Extract features for clustering
props = measure.regionprops(label_im)
axlength = [(prop.major_axis_length, prop.minor_axis_length) for prop in props]
axlength = np.array(axlength)
fig, ax = plt.subplots()
ax.scatter(axlength[:, 0], axlength[:, 1])
ax.set_xlabel('Major axis length')
ax.set_ylabel('Minor axis length')
fname = 'scatterplot_axlength.png'
fig.savefig(fname)
fname
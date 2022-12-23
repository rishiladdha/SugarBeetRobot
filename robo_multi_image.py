
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



# %% process all the images

props_list = []
im_idx=[]
axes=[]
for i in range(len(nir_list)):
    nir = nir_list[i]
    thresh = filters.threshold_otsu(nir)
    binary = nir > thresh
    morphed = morphology.remove_small_objects(binary, 100)
    morphed = morphology.binary_erosion(morphed, morphology.disk(2))
    morphed = morphology.binary_dilation(morphed, morphology.disk(2))
    morphed = morphology.binary_opening(morphed, footprint=morphology.diamond(5))
    morphed = morphology.binary_closing(morphed, morphology.disk(2))
    label_im = measure.label(morphed, background=0)
    props = measure.regionprops(label_im)
    props_list += props
    axlength = [(prop.major_axis_length, prop.minor_axis_length, prop.area) for prop in props]
    for prop in props:
        im_idx.append(i)
    axlength = np.array(axlength)
    axes.append(axlength)
    fanme = '/Users/laddharishi/PycharmProjects/DSAI/DS Assignments End Sem/Morphed Images/morphed_image'+str(i)+'.png'
    skio.imsave(fanme, morphed)


# %% plot feature matrix for all images

fig, ax = plt.subplots()
for i in range(len(axes)):
    ax.scatter(axes[i][:, 0], axes[i][:, 1], color = 'orange')
ax.set_xlabel('Major axis length')
ax.set_ylabel('Minor axis length')
fname = 'feature_matrix_for_all_images.png'
fig.suptitle('Feature matrix for all images', fontdict={'color': 'blue'})
fig.savefig(fname)
fname

#%% Clustering
feature_matrix = np.concatenate(axes)
nc = 4
kmeans = cluster.KMeans(init='k-means++', n_clusters=nc, n_init=3, random_state=0)
kmeans.fit(feature_matrix[:,:2])
labels = kmeans.predict(feature_matrix[:,:2])
fig, ax = plt.subplots()
for label in range(nc):
    ax.scatter(feature_matrix[:, 0][labels == label], feature_matrix[:, 1][labels == label])
fig.suptitle('Clustered data (minor and major axes)', fontdict={'color': 'blue'})
fname = 'clusters_1.png'
fig.savefig(fname)
fname

#%% Clustering with area
nc = 4
kmeans = cluster.KMeans(init='k-means++', n_clusters=nc,
n_init=3, random_state=0)
kmeans.fit(feature_matrix)
labels = kmeans.predict(feature_matrix)
fig, ax = plt.subplots()
for label in range(nc):
    ax.scatter(feature_matrix[:, 0][labels == label], feature_matrix[:, 1][labels == label])
fig.suptitle('Clustered data (minor and major axes and area)', fontdict={'color': 'blue'})
fname = 'clusters_2.png'
fig.savefig(fname)
fname

#%% Check clusters in some random images
# Turning these into numpy arrays makes indexing simpler
im_idx = np.array(im_idx)
props_list = np.array(props_list)
idx = np.random.choice(len(nir_list)) # pick a random index
nir = nir_list[idx] # image to check
# feature matrix rows corresponding to image with index idx
fidx = np.flatnonzero(im_idx == idx)
fig, ax = plt.subplots()
ax.imshow(nir, cmap='gray')
# 3 colors for 3 clusters
colors = ['orange', 'magenta', 'blue', 'yellow']
# Remember this from before??
for index in fidx:
    prop = props_list[index]
    label = labels[index]
    y0, x0, y1, x1 = prop.bbox
    plt.plot((x0, x1, x1, x0, x0), (y0, y0, y1, y1, y0), '-', color=colors[label])
    y0, x0 = prop.centroid
    x1 = x0 + np.cos(prop.orientation) * 0.5 * prop.minor_axis_length
    y1 = y0 - np.sin(prop.orientation) * 0.5 * prop.minor_axis_length
    x2 = x0 - np.sin(prop.orientation) * 0.5 * prop.major_axis_length
    y2 = y0 - np.cos(prop.orientation) * 0.5 * prop.major_axis_length
    ax.plot((x0, x1), (y0, y1), '-', color=colors[label])
    ax.plot((x0, x2), (y0, y2), '-', color=colors[label])
fig.suptitle('Labeled data', fontdict={'color': 'blue'})
fname = 'clusters_check.png'
fig.savefig(fname)
fname

# %% compute the mean and standard deviation of major and minor axis length and blob area for each cluster
mean_major=[]
mean_minor=[]
mean_area=[]
std_major=[]
std_minor=[]
std_area=[]


for i in range(nc):
    mean_major.append(np.mean(feature_matrix[:,0][labels == i]))  
    mean_minor.append(np.mean(feature_matrix[:,1][labels == i]))
    mean_area.append(np.mean(feature_matrix[:,2][labels == i]))
    std_major.append(np.std(feature_matrix[:,0][labels == i]))
    std_minor.append(np.std(feature_matrix[:,1][labels == i]))
    std_area.append(np.std(feature_matrix[:,2][labels == i]))
    
#print(mean_major, mean_minor, mean_area)

# %%
p_value = scipy.stats.ttest_ind(feature_matrix[:,2][labels == 0], feature_matrix[:,2][labels == 2], equal_var=False)
print(p_value)
#concluding, because p-value is less than 0.05, we can conclude that the two clusters are significantly different.



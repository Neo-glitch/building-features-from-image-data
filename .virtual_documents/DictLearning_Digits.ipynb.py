import skimage
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()


digits.data.shape


# number of image to work with since dict learnin takes time
NUM_IMAGES = 38

sample = digits.data[:NUM_IMAGES, :]


sample.shape # mean 38 images of 8 x 8 size


from sklearn.decomposition import DictionaryLearning

# dictionary learning, reduce component from 64 to 35 and it takes time to run
dict_learn = DictionaryLearning(n_components = 36, fit_algorithm="lars",
                               transform_algorithm="lasso_lars")

X_dict = dict_learn.fit_transform(sample)


X_dict.shape


# resize X-dict image for plotting in matplot
from skimage import transform

resized = transform.resize(X_dict[0].reshape(6, 6), (8,8))

plt.figure(figsize = (6, 6))
plt.imshow(resized)


# sparse coding for all sample images
fig = plt.figure(figsize = (10, 10))

for i in range(NUM_IMAGES):
    ax = fig.add_subplot(6, 8, i+1, xticks = [], yticks = [])
    ax.imshow(X_dict[i].reshape(6, 6), cmap = "Blues_r", interpolation = "nearest")


# get the original image back(sparse rep x dict atoms extracted)
import numpy as np

original = np.matmul(X_dict, dict_learn.components_)


plt.imshow(original[2].reshape(8, 8), cmap = "Blues_r")



















































































































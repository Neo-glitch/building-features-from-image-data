import tensorflow.keras as keras
import gzip
import numpy as np
import matplotlib.pyplot as plt


# image feature, since each image has this 28 * 28
IMAGE_FEATURES = 28 * 28 


# helper fun to xtract images from gzip file
def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_FEATURES * num_images)  # specifies num of bytes to read
        
        # convert buffer to image matrix of type float
        data = np.frombuffer(buf, dtype = np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_FEATURES)
        
        return data


train_data = extract_data("./datasets/train-images-idx3-ubyte.gz", 60000)
test_data = extract_data("./datasets/t10k-images-idx3-ubyte.gz", 10000)


train_data.shape, test_data.shape


# labels
label_dict = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot"
}


# helper to show just one image
def display_image(image_pixels):
    plt.imshow(image_pixels.reshape(28, 28), cmap = "gray")


display_image(train_data[50])


# calc max and min pixel value of imag data
np.max(train_data), np.min(train_data)


# rescale pixel value to 0 -1 
train_data /= 255.0
test_data /= 255.0


np.max(train_data), np.min(train_data)


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense


batch_size = 128
epochs = 1


# SHAPE OF EVERY INPUT 784
input_img = Input(shape = (IMAGE_FEATURES, ))


# helper fn to create autoencoder(input layer size == output layer size and same as input data size)
def autoencoder(input_img):
    encoding1 = Dense(784, activation = "relu")(input_img)
    encoding2 = Dense(256, activation = "relu")(encoding1)
    
    # the hidden or coding layer
    coding = Dense(64, activation = "relu")(encoding2)
    
    decoding2 = Dense(256, activation = "relu")(coding)
    decoding1 = Dense(784, activation = "relu")(decoding2)
    
    return decoding1


autoencoder = Model(input_img, autoencoder(input_img))

autoencoder.compile(loss = "mean_squared_error", optimizer = Adam())


autoencoder.summary()


autoencoder_train = autoencoder.fit(train_data, train_data,
                                   batch_size=batch_size,
                                   epochs = epochs,
                                   verbose = 1,
                                   validation_data = (train_data, train_data))


# use autoencoder to predict test data, try to reconstruct test data using features learnt
pred = autoencoder.predict(test_data)

pred.shape


# check if autoencoder reconstructed images well
plt.figure(figsize = (24, 24))
print("Original")


# iter through test data and plot using plt
pos = 0
for i in range(100, 107):
    plt.subplot(2, 10, pos+1)
    
    img = test_data[i].reshape(28, 28)
    plt.imshow(img, cmap = "gray")
    pos = pos + 1
    
plt.show()


# iter through test data reconstructed by autoencoder.
plt.figure(figsize = (24, 24))
print("Reconstruction")

pos = 0
for i in range(100, 107):
    plt.subplot(2, 10, pos+1)
    
    img = pred[i].reshape(28, 28)
    plt.imshow(img, cmap = "gray")
    pos = pos + 1
    
plt.show()













































































































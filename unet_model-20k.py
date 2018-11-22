def setup_tensorflow_theano(cpu=False):
   from keras import backend as K

   if 'tensorflow' != K.backend():
       return
   K.clear_session()

   import tensorflow as tf
   from keras.backend.tensorflow_backend import set_session
   if cpu:
       config = tf.ConfigProto(device_count={'GPU': 0})
   else:
       config = tf.ConfigProto()
   config.gpu_options.allow_growth = True
   set_session(tf.Session(config=config))

setup_tensorflow_theano()


#
#   Import the npy files with images and labels
#
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as  np
images = np.load('train_images.npy')
labels = np.load('train_labels.npy')

# taking the image size to create the model
image_rows, image_cols, image_channels = images[0].shape

#
#   U-net
#
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, merge, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

input_layer = Input((image_rows, image_cols, image_channels))

conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

up6 = UpSampling2D(size=(2, 2))(conv5)
up6 = merge.concatenate([up6, conv4], axis=3)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

up7 = UpSampling2D(size=(2, 2))(conv6)
up7 = merge.concatenate([up7, conv3], axis=3)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

up8 = UpSampling2D(size=(2, 2))(conv7) 
up8 = merge.concatenate([up8, conv2], axis=3)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

up9 = UpSampling2D(size=(2, 2))(conv8) 
up9 = merge.concatenate([up9, conv1], axis=3)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

flatten = Flatten()(conv10)
Dense1 = Dense(512, activation='relu')(flatten)
Dense2 = Dense(24, activation='linear')(Dense1)
    

#Create the model
model = Model(inputs=[input_layer], outputs=[Dense2])

opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mse'])
model.summary()

# Set callback functions to early stop training
callbacks = [EarlyStopping(monitor='val_loss', patience=5), ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

# Fit for the 10k images
results = model.fit(images[:100], labels[:100], batch_size=1, epochs=5000, callbacks=callbacks, validation_split=0.2)
model.save('model-test-early.h5')

# Predictions
predictions = model.predict(images[:100])
predictions = abs(np.rint(predictions))
np.save('predictions', predictions)

# Get training Acc
import numpy as np
num_match=0
for i in range(len(labels[:100])):
    if np.array_equal(labels[i], predictions[i]):
        num_match+=1
    

acc = num_match / len(labels[:100])
print(" Train Accuracy: {0:.2f}%".format(acc*100))




##
##   Random Samples
##
#import random
#import matplotlib.pyplot as plt
#random_samples = random.sample(range(10000), 3)
#    
#for i in range(len(random_samples)):  
#    random_predictions = predictions[random_samples[i]]
#    true = labels[random_samples[i]]
#    print('Prediction {0}: {1}'.format(random_samples[i], predictions[random_samples[i]]))
#    print('True {0}: {1}: '.format(random_samples[i], labels[random_samples[i]]))
#    plt.imshow(images[random_samples[i]])
#    plt.savefig('images10k-' + str(random_samples[i]) + '.png')

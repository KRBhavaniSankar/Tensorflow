
# -*- coding: utf-8 -*-
"""
Created on WED May 09 11:52:10 2018
@author: Bhavani
"""

"""
Sequential  : Initialize our neural network model as a sequential network.  There are two basic ways of initializing a neuarl network , either by a sequence of layers or as a graph.
Conv2D      : First step of CNN is convolution operation on the training images here, which is basically 2D arrays , we're using a convolution 2-D , you may have to use Convulation 3-D while dealing with videos , where the third dimension will be the time.
Maxpooling2D:  that is the step — 2 in the process of building a cnn. For building this particular neural network, we are using a Maxpooling function, there exist different types of pooling operations like Min Pooling, Mean Pooling, etc. Here in MaxPooling we need the maximum value pixel from the respective region of interest
Flatten     : Used for flattening, it is the process of converting all the resultant 2 dimensional arrays into a single long contineous linear vector.
Dense       :This is used to perform the full connection of the neural network, which is the step 4 in the process of building a CNN.
"""


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from PIL import Image

#Now, we will create an object of the sequential class below:

classifier = Sequential()
classifier.add(Conv2D(32,(3,3), input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())

'''
In this step we need to create a fully connected layer, and to this layer we are going to connect the set of nodes we got after the flattening step, these nodes will act as an input layer to these fully-connected layers. As this layer will be present between the input layer and output layer, we can refer to it a hidden layer.
'''
classifier.add(Dense(units = 128, activation = 'relu'))

'''
Now it’s time to initialise our output layer, which should contain only one node, as it is binary classification. This single node will give us a binary output of either a Cat or Dog.
'''
classifier.add(Dense(units=1,activation='sigmoid'))

#Now that we have completed building our CNN model, its time to compile it.

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
#Image pre processing
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('/home/bhavani/work/Python/programs/DL/CNN_Data/training_set',
                                                 target_size=(64,64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('/home/bhavani/work/Python/programs/DL/CNN_Data/test_set',
                                            target_size=(64,64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=2000)


#Making new predictions on our trained model:
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/home/bhavani/work/Python/programs/DL/CNN_Data/test_set/cats/cat.4011.jpg',
                            target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction='cat'

print(prediction)


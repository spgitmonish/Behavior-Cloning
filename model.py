import random
import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Keras Libraries
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam

# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

# Hyperparameters
# Definitely affects performance, defaults to 1.0 but changed during training
global_keep_set_threshold = 1.0
batch_size = 128
number_of_epochs = 8

def preprocessImageAndSteering(driving_data_df):
    '''
    This function preprocesses the images before being fed into the model.
    The caller of this function is a custom generator
    '''
    # Pick an image from the center, left or right
    camera_angle_to_pick = np.random.randint(3)

    # One of the lists has to have a match
    # NOTE: There will be only one hit per image, so the list has only one element
    if camera_angle_to_pick == 0:
        # Get the steering angle of the corresponding image
        steering_angle = driving_data_df["SteeringAngle"][0]
        file_path = str(driving_data_df["Center"][0].strip())
    elif camera_angle_to_pick == 1:
        steering_angle = driving_data_df["SteeringAngle"][0]
        # The aim of the model is to recenter the car if it the view is
        # not completely of the lane, here the left camera's corresponding
        # steering angle is compensated
        steering_angle = steering_angle + 0.20
        file_path = str(driving_data_df["Left"][0].strip())
    elif camera_angle_to_pick == 2:
        # Same logic but for the right camera
        steering_angle = driving_data_df["SteeringAngle"][0]
        steering_angle = steering_angle - 0.25
        file_path = str(driving_data_df["Right"][0].strip())

    # Get current working directory
    cwd = os.getcwd()

    # Path to images
    images_path = cwd + "/IMG"
    image = cv2.imread(os.path.join(images_path, file_path), cv2.IMREAD_COLOR)

    # Images are read in BGR format with openCV, convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # The Nvidia model uses YUV format as it's input
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    # Crop out bottom 25 pixels of the image
    image = image[0:image.shape[0]-25, 0:image.shape[1]]

    # Resize to have 200 cols and 64 rows
    image = cv2.resize(image, (200, 66))
    image = np.array(image)

    return image, steering_angle

def customImageGenerator(driving_data_frame, batch_size = 128):
    '''
    Custom generator for generating images on the fly, instead of
    storing the files in memory.
    '''
    # Nvidia model is designed for images which are in the form
    # 200(width)x66(height)x3(channels)
    X_data = np.zeros((batch_size, 66, 200, 3))
    Y_data = np.zeros(batch_size)

    # Generates batches of augmented/normalized data.
    # Yields batches indefinitely, in an infinite loop.
    # This is similar to flow() method of the ImageDataGenerator object
    while 1:
        for batch_index in range(batch_size):
            # Get a random index
            image_index = np.random.randint(len(driving_data_frame))
            # Get the corresponding data
            data_frame = driving_data_frame.iloc[[image_index]].reset_index()

            # Flag which enables weeding out the lower steering angle data
            get_new_set = True

            # Setting the flag above to false, disables weeding out.
            # NOTE: This mode hasn't worked, so recommend turning it off
            if get_new_set == False:
                X, Y = preprocessImageAndSteering(data_frame)

            # Source: http://bit.ly/2jzjaOq, Author: Vivek Yadav
            while get_new_set == True:
                # Pre process the image
                X, Y = preprocessImageAndSteering(data_frame)

                # To reduce bias towards driving straight
                # don't add images to the set which have steering angle < 0.1
                if abs(Y) < 0.1:
                    # Get a value in the range 0.0-1.0
                    keep_set_probability = np.random.uniform()

                    # Idea behind this is to initially weed out lower steering
                    # angle data aggressively at the beginning of training and
                    # then ease off by checking the threshold. In my case this
                    # is actually a hyperparameter which affected my performance.
                    if keep_set_probability > global_keep_set_threshold:
                        get_new_set = False
                else:
                    get_new_set = False

            # Append to the current batch
            X_data[batch_index] = X
            Y_data[batch_index] = Y

        # yield is like return but will return a generator here
        yield X_data, Y_data

def normalizeImage(image):
    '''
        Normalize image in the range -0.5 + 0.5
    '''
    image = image/255.-.5
    return image

def createNVidiaModel():
    '''
        Using Keras Sequential API the Nvidia model is created with:
        1. 3 Convolution layers with 2x2 strides, 5x5 kernels(24, 36, 48 filters)
        2. 2 Convolution layes with unit strides and 3x3 kernels(64 & 64 filters)
        3. 3 Fully Connected Layers with 100, 50, 10 outputs
        4. Final output layer which predicts the steering angle
    '''
    model = Sequential()

    # Add a Lambda layer for normalization of the input image
    model.add(Lambda(normalizeImage, input_shape=(66, 200, 3)))

    # Layer 1: Convolution with 24 Filters and a 5x5 kernel
    # NOTE: The input shape needs to defined only for the 1st layer
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2)))
    # Layer 1: ELU Activation
    model.add(ELU())
    # Layer 1: Dropout 
    model.add(Dropout(0.25))

    # Layer 2: Convolution with 36 Filters and a 5x5 kernel
    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2)))
    # Layer 2: ELU Activation
    model.add(ELU())
    # Layer 2: Dropout 
    model.add(Dropout(0.25))

    # Layer 3: Convolution with 48 Filters and 5x5 kernel
    model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2)))
    # Layer 3: ELU Activation
    model.add(ELU())
    # Layer 3: Dropout 
    model.add(Dropout(0.25))

    # Layer 4: Convolution with 64 Filters and a 3x3 kernel
    # NOTE: Weight regularization applied(here and next) to prevent overfitting
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), W_regularizer=l2(0.01)))
    # Layer 4: ELU Activation
    model.add(ELU())
    # Layer 4: Dropout 
    model.add(Dropout(0.25))

    # Layer 5: Convolution with 64 Filters and a 3x3 kernel
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), W_regularizer=l2(0.01)))
    # Layer 5: ELU Activation
    model.add(ELU())
    # Layer 5: Dropout 
    model.add(Dropout(0.25))

    # Flatten the output of the previous layer(1164 outputs from previous layer)
    model.add(Flatten())
    model.add(ELU())
    model.add(Dropout(0.25))

    # Layer 6: Fully Connected layer with 100 outputs
    model.add(Dense(100))
    # Layer 6: ELU activation
    model.add(ELU())

    # Layer 7: Fully Connected layer with 50 outputs
    model.add(Dense(50))
    # Layer 7: ELU activation
    model.add(ELU())

    # Layer 8: Fully Connected layer with 10 outputs
    model.add(Dense(10))
    # Layer 8: ELU activation
    model.add(ELU())

    # Output layer which predicts a value
    model.add(Dense(1))

    # NOTE: No activation is used because this is a numerical computation
    return model

def createAndTrainModel():
    '''
        Function creates, compiles and runs the model for training
        NOTE: There is no split of data for testing, the testing is
              on the simulator, because low loss doesn't hold value
              on the test set as a high accuracy(low loss) doesn't
              necessarily mean better performance during simulation
    '''
    # Compile and train the model
    model = createNVidiaModel()

    # Configure the model to apply Adam's SGD to minimize the mean squared error
    # NOTE: The learning rate for adam optimizer is default set to 0.001
    model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')

    # Read the csv file into a data frame
    driving_log_df = pd.read_csv('driving_log.csv')

    # Create validation test from training set
    train_df, validation_df = train_test_split(driving_log_df, test_size=0.2)

    # Training data generator
    training_data_generator = customImageGenerator(train_df, batch_size)

    # Validation data generator
    validation_data_generator = customImageGenerator(validation_df, batch_size)

    # Fit the model on the data generated batch-by-batch by the generator.
    # NOTE from Keras: This allows real-time data augmentation on images on CPU
    #                  in parallel to training your model on GPU.
    global_keep_set_threshold = 0.5
    for epoch in range(1, (number_of_epochs + 1)):
        # Print global threshold
        print("Threshold:", global_keep_set_threshold)

        model.fit_generator(training_data_generator,
                            samples_per_epoch=20000,
                            validation_data=validation_data_generator,
                            nb_val_samples=2000,
                            nb_epoch=1)

        # Threshold to determine to keep an set or not
        global_keep_set_threshold = global_keep_set_threshold/(1 + epoch)

    return model

def saveModel(model):
    '''
        Saves the model created in the current working directory
    '''
    model_json = model.to_json()
    open('model.json', 'w').write(model_json)
    model.save_weights('model.h5', overwrite=True)

# Train and save the model
model = createAndTrainModel()
saveModel(model)

# Optionally look at the summary
# Nice way to sanity check the input, output and parameter sizes
# model.summary()

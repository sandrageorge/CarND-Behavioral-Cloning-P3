import csv
import numpy as np
from click.core import batch
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Dropout, Cropping2D, Activation
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
import cv2

# lines = []
# firstline = True
# with open('data/data/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     for line in reader:
#         if firstline:  # skip first line
#             firstline = False
#             continue
#         lines.append(line)
#
# images = []
# measurements = []
#
# for line in lines:
#     source_path = line[0]
#     filename = source_path.split('/')[-1]
#     current_path = 'data/data/IMG/' + filename
#     image = cv2.imread(current_path)
#     images.append(image)
#
#     measurement = float(line[3])
#     measurements.append(measurement)
#
# X_train = np.array(images)
# y_train = np.array(measurements)


def generate_samples(paths):
    samples = []
    for path in paths:
        csv_path = path + 'driving_log.csv'
        with open(csv_path) as csvfile:
            reader = csv.reader(csvfile)
            for i, line in enumerate(reader):
                if i > 0:
                    samples.append(line)

    print("Testing data number:", len(samples))

    return samples


def generator(samples, split_str, batch_size_=64):
    correction = 0.2
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size_):
            batch_samples = samples[offset:offset + batch_size_]

            images = []
            angles = []
            for batch_sample in batch_samples:
                source_path = batch_sample[0]
                current_path = '..' + source_path.split(split_str)[-1].replace('\\', '/')
                # current_path = '../track1/data/data/' + source_path

                image = cv2.imread(current_path)
                angle = float(batch_sample[3])
                images.append(image)
                angles.append(angle)

                source_path = batch_sample[1]
                current_path = '..' + source_path.split(split_str)[-1].replace('\\', '/')
                # current_path = '../track1/data/data/' + source_path

                image = cv2.imread(current_path)
                angle = float(batch_sample[3])
                images.append(image)
                angles.append(angle+0.2)

                source_path = batch_sample[2]
                current_path = '..' + source_path.split(split_str)[-1].replace('\\', '/')
                # current_path = '../track1/data/data/' + source_path

                image = cv2.imread(current_path)
                angle = float(batch_sample[3])
                images.append(image)
                angles.append(angle-0.2)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

def InitialModel():
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0, input_shape=(160,320,3)))
    model.add(Flatten())
    model.add(Dense(1))

    return model

def LeNet():
    leNetModel = Sequential()
    #LeNet.add(Cropping2D(cropping=((40,20),(0,0)), input_shape=(160,320,3)))
    leNetModel.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

    leNetModel.add(Convolution2D(6, 5, 5, subsample=(1, 1), border_mode='valid', activation='elu'))
    leNetModel.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), border_mode='valid'))

    leNetModel.add(Convolution2D(16, 5, 5, subsample=(1, 1), border_mode="valid", activation='elu'))
    leNetModel.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), border_mode='valid'))

    leNetModel.add(Flatten())

    leNetModel.add(Dropout(0.5))
    leNetModel.add(Dense(120, activation='elu'))

    leNetModel.add(Dropout(0.5))
    leNetModel.add(Dense(84, activation='elu'))

    leNetModel.add(Dense(10, activation='elu'))

    leNetModel.add(Dense(1))
    return leNetModel

def Nvidia(input_shape=(160, 320, 3)):
    nVidia = Sequential()
    nVidia.add(Cropping2D(cropping=((40,20),(0,0)), input_shape=(160,320,3)))
    nVidia.add(Lambda(lambda x: x / 255.0 - 0.5, name="image_normalization", input_shape=input_shape))

    nVidia.add(Convolution2D(24, 5, 5, name="convolution_1", subsample=(2, 2), border_mode="valid", init='he_normal'))
    nVidia.add(ELU())
    nVidia.add(Convolution2D(36, 5, 5, name="convolution_2", subsample=(2, 2), border_mode="valid", init='he_normal'))
    nVidia.add(ELU())
    nVidia.add(Convolution2D(48, 5, 5, name="convolution_3", subsample=(2, 2), border_mode="valid", init='he_normal'))
    nVidia.add(ELU())
    nVidia.add(Convolution2D(64, 3, 3, name="convolution_4", border_mode="valid", init='he_normal'))
    nVidia.add(ELU())

    nVidia.add(Convolution2D(64, 3, 3, name="convolution_5", border_mode="valid", init='he_normal'))
    nVidia.add(ELU())
    nVidia.add(Flatten())

    nVidia.add(Dropout(0.5))

    nVidia.add(Dense(100, name="hidden1", init='he_normal'))
    nVidia.add(ELU())

    nVidia.add(Dropout(0.5))

    nVidia.add(Dense(50, name="hidden2", init='he_normal'))
    nVidia.add(ELU())
    nVidia.add(Dense(10, name="hidden3", init='he_normal'))
    nVidia.add(ELU())

    nVidia.add(Dense(1, name="steering_angle", activation="linear"))

    return nVidia

epochs = 30
batch_size = 8
model = Nvidia(input_shape=(160, 320, 3))
#model = LeNet()
# model = InitialModel()

# model.compile(optimizer=Adam(lr=0.001), loss='mse')
# # model.summary()
# # model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=batch_size,  validation_split=0.2, shuffle=True)
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=epochs)

# paths, split_str = ['../track1/2labs_center/', '../track1/1lab_recovery/', '../track1/1lab_smothcurve/', '../track1/FirstRecord/', '../track1/Recover/', '../track1/Recover2/'], 'Behavioral_cloning'
paths, split_str = ['../track1/data/data/'], 'Behavioral_cloning'
samples = generate_samples(paths)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
path = []
train_generator = generator(train_samples, split_str=split_str, batch_size_=batch_size)
validation_generator = generator(validation_samples, split_str=split_str, batch_size_=batch_size)

model.compile(optimizer=Adam(lr=0.001), loss='mse')
history_object = model.fit_generator(train_generator,
                                     steps_per_epoch=len(train_samples) / batch_size,
                                     validation_data=validation_generator,
                                     validation_steps=len(validation_samples) / batch_size,
                                     epochs=epochs)



model.save('models/nVidia_side_records.h5')

# plot the training and validation loss for each epoch
from matplotlib import pyplot as plt
print("Training loss: ", history_object.history['loss'])
print("Validation loss: ", history_object.history['val_loss'])
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
import csv
import numpy as np
from click.core import batch
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Dropout, Cropping2D, Conv2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
import cv2

def generate_samples(paths):
    samples = []
    for path in paths:
        csv_path = path + 'driving_log.csv'
        with open(csv_path) as csvfile:
            reader = csv.reader(csvfile)
            for i, line in enumerate(reader):
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
                current_path = '../../' + source_path.split(split_str)[-1].replace('\\', '/')
                # current_path = '../track1/data/data/' + source_path

                image = cv2.imread(current_path)
                angle = float(batch_sample[3])
                images.append(image)
                angles.append(angle)

                source_path = batch_sample[1]
                current_path = '../../' + source_path.split(split_str)[-1].replace('\\', '/')
                # current_path = '../track1/data/data/' + source_path

                image = cv2.imread(current_path)
                angle = float(batch_sample[3])
                images.append(image)
                angles.append(angle+correction)

                source_path = batch_sample[2]
                current_path = '../../' + source_path.split(split_str)[-1].replace('\\', '/')
                # current_path = '../track1/data/data/' + source_path

                image = cv2.imread(current_path)
                angle = float(batch_sample[3])
                images.append(image)
                angles.append(angle-correction)

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

def Nvidia():
    model = Sequential()
    model.add(Cropping2D(cropping=((40, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(64, 64, 3)))
    model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

epochs = 30
batch_size = 8
model = Nvidia()
#model = LeNet()
# model = InitialModel()

# model.compile(optimizer=Adam(lr=0.001), loss='mse')
# # model.summary()
# # model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=batch_size,  validation_split=0.2, shuffle=True)
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=epochs)

# paths, split_str = ['../track1/2labs_center/', '../track1/1lab_recovery/', '../track1/1lab_smothcurve/', '../track1/FirstRecord/', '../track1/Recover/', '../track1/Recover2/'], 'Behavioral_cloning'
paths, split_str = ['../../track1/data/'], '07_Behavioral_Cloning'
samples = generate_samples(paths)
train_samples, validation_samples = train_test_split(samples, test_size=0.15)
path = []
train_generator = generator(train_samples, split_str=split_str, batch_size_=batch_size)
validation_generator = generator(validation_samples, split_str=split_str, batch_size_=batch_size)

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
                                     steps_per_epoch=len(train_samples) / batch_size,
                                     validation_data=validation_generator,
                                     validation_steps=len(validation_samples) / batch_size,
                                     epochs=epochs)



save_str = "Nvidia" + '_e' + str(epochs) + '_b' + str(batch_size) + '.h5'
model.save(save_str)
print(save_str)

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
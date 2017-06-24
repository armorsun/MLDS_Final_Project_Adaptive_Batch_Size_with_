from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import adam
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
import keras.callbacks
import argparse
import numpy
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import csv
import math
from keras.datasets import mnist, cifar10

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

parser = argparse.ArgumentParser()
parser.add_argument('--batch_type', nargs='+', default='constant', help='constant, linear, exp, gau')
parser.add_argument('--dataset', nargs=1, help='mnist, cifar10, fish')
args = vars(parser.parse_args())
training_batch_type = args['batch_type'][0]
dataset = args['dataset'][0]

seed = 7
numpy.random.seed(seed)

print('training on %s dataset, batch type is %s' % (dataset, training_batch_type))

if dataset == 'mnist':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    img_width, img_height = 28, 28
    num_class = 10
    batch_size = int(args['batch_type'][1])
    num_epoch = 20
    num_training = x_train.shape[0]

    x_train = x_train.reshape(x_train.shape[0], img_width, img_height, 1)
    x_test = x_test.reshape(x_test.shape[0], img_width, img_height, 1)
    input_shape = (img_width, img_height, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, num_class)
    y_test = keras.utils.to_categorical(y_test, num_class)

elif dataset == 'cifar10':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    img_width, img_height = 32, 32
    num_class = 10
    batch_size = int(args['batch_type'][1])
    num_epoch = 20
    num_training = x_train.shape[0]

    input_shape = (img_width, img_height, 3)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, num_class)
    y_test = keras.utils.to_categorical(y_test, num_class)

elif dataset == 'fish':

    training_data_dir = 'datasets/train'
    test_data_dir = 'datasets/test'

    img_width, img_height = 300, 100
    input_shape = (img_width, img_height, 3)
    num_training = 3760
    num_test = 940
    num_epoch = 20
    num_class = 3
    batch_size = int(args['batch_type'][1])

name = 'models/model2' + dataset + '_' + training_batch_type + '_bs' + args['batch_type'][1]

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(num_class, activation='softmax'))

#adam = adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
print(model.summary())

tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

loss_save = list();
acc_save = list();
val_acc_save = list();
val_loss_save = list();
batch_num_save = list();

for num_epoch_now in range(num_epoch):

    if training_batch_type == 'constant':
        batch_size_d = batch_size
    elif training_batch_type == 'linear':
        epoch_batch_rate = float(batch_size - 32) / float(num_epoch - 1)
        batch_size_d = int(round(batch_size - num_epoch_now * epoch_batch_rate))
    elif training_batch_type == 'cos':
        batch_size_d = int(round(224 * math.cos(4.5 * num_epoch_now * math.pi / 180))) + 32
    elif training_batch_type == 'gau':
        batch_size_d = int(round(((batch_size - 32) * math.exp(-(float(num_epoch_now) / 10) ** 2 / 0.04)))) + 32
    elif training_batch_type == 'exp':
        batch_size_d = int(round((batch_size - 32) / ((float(num_epoch_now + 3) / 3) ** 2))) + 32

    print('batch size: ', batch_size_d)
    print('epoch: ', num_epoch_now + 1)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    if dataset == 'mnist' or dataset == 'cifar10':
        train_datagen.fit(x_train)
        train_generator = train_datagen.flow(
            x_train,
            y_train,
            batch_size=batch_size_d
        )

        his = model.fit_generator(
            train_generator,
            steps_per_epoch=round(num_training / batch_size_d),
            epochs=1,
            callbacks=[tensor_board],
            validation_data=(x_test, y_test)
        )

    elif dataset == 'fish':
        train_generator = train_datagen.flow_from_directory(
            training_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size_d,
            class_mode='categorical'
        )

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        validation_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size_d,
            class_mode='categorical'
        )

        his = model.fit_generator(
            train_generator,
            steps_per_epoch=round(num_training / batch_size_d),
            epochs=1,
            callbacks=[tensor_board],
            validation_data=validation_generator,
            validation_steps=num_test
        )

    loss_save.append(his.history['loss'][0])
    acc_save.append(his.history['acc'][0])
    val_loss_save.append(his.history['val_loss'][0])
    val_acc_save.append(his.history['val_acc'][0])
    batch_num_save.append(batch_size_d)

with open(name + '_loss.csv', 'w') as csvfile:
    loss_writer = csv.writer(csvfile)
    loss_writer.writerow(loss_save)

with open(name + '_acc.csv', 'w') as csvfile:
    acc_writer = csv.writer(csvfile)
    acc_writer.writerow(acc_save)

with open(name + '_val_acc.csv', 'w') as csvfile:
    val_acc_writer = csv.writer(csvfile)
    val_acc_writer.writerow(val_acc_save)

with open(name + '_val_loss.csv', 'w') as csvfile:
    val_loss_writer = csv.writer(csvfile)
    val_loss_writer.writerow(val_loss_save)

with open(name + '_batch_num.csv', 'w') as csvfile:
    batch_writer = csv.writer(csvfile)
    batch_writer.writerow(batch_num_save)

model_json = model.to_json()
with open(name + ".json", "w") as json_file: json_file.write(model_json)
model.save_weights(name + '.h5')

print(name)

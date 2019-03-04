import keras
import numpy as np
import keras.backend as K
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.models import Sequential
from keras.models import load_model
from keras import regularizers, optimizers
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import glorot_normal, RandomNormal, Zeros
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)

num_classes = 10
y_train = np_utils.to_categorical(y_train,num_classes)
y_test = np_utils.to_categorical(y_test,num_classes)



datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
    vertical_flip=True
    )
datagen.fit(x_train)


def create_model(s = 2, weight_decay = 1e-2, act="relu"):
    model = Sequential()


    model.add(Conv2D(64, (3,3), padding='same', kernel_initializer=glorot_normal(), 			input_shape=x_train.shape[1:]))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
   
    model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=glorot_normal()))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    
    model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=RandomNormal(stddev=0.01)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    

    model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=RandomNormal(stddev=0.01)))
    model.add(BatchNormalization())
    model.add(Activation(act))


    model.add(MaxPooling2D(pool_size=(2,2), strides=s))
    model.add(Dropout(0.2))
    

    model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=RandomNormal(stddev=0.01)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    

    model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=glorot_normal()))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    

    model.add(Conv2D(256, (3,3), padding='same', kernel_initializer=glorot_normal()))

    model.add(MaxPooling2D(pool_size=(2,2), strides=s))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    

    model.add(Conv2D(256, (3,3), padding='same', kernel_initializer=glorot_normal()))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    

    model.add(Conv2D(256, (3,3), padding='same', kernel_initializer=glorot_normal()))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))

    model.add(MaxPooling2D(pool_size=(2,2), strides=s))
    
    

    model.add(Conv2D(512, (3,3), padding='same', kernel_initializer=glorot_normal()))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))


    model.add(Conv2D(2048, (1,1), padding='same', kernel_initializer=glorot_normal()))
    model.add(Activation(act))
    model.add(Dropout(0.2))
    

    model.add(Conv2D(256, (1,1), padding='same', kernel_initializer=glorot_normal()))
    model.add(Activation(act))

    model.add(MaxPooling2D(pool_size=(2,2), strides=s))
    model.add(Dropout(0.2))



    model.add(Conv2D(256, (3,3), padding='same', kernel_initializer=glorot_normal()))
    model.add(Activation(act))

    model.add(MaxPooling2D(pool_size=(2,2), strides=s))


    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    return model

if __name__ == "__main__":
    model = create_model(act="relu")
    batch_size = 128
    epochs = 100
    train = {}


    opt_adm = keras.optimizers.Adadelta(lr= 0.8, rho=0.95)
    model.compile(loss='categorical_crossentropy', optimizer=opt_adm, metrics=['accuracy'])
    train["part_1"] = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                        steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,
                                        verbose=1,validation_data=(x_test,y_test))
    model.save("simplenet_generic_first.h5")
    print(train["part_1"].history)

    
    opt_adm = keras.optimizers.Adadelta(lr=0.7, rho=0.9, decay = 1e-5)
    model.compile(loss='categorical_crossentropy', optimizer=opt_adm, metrics=['accuracy'])
    train["part_2"] = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                        steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,
                                        verbose=1,validation_data=(x_test,y_test))
    model.save("simplenet_generic_second.h5")
    print(train["part_2"].history)



    print("\n \n Final Logs: ", train)


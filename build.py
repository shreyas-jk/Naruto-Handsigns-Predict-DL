from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import optimizers
import json

image_width, image_height = 64, 64
if K.image_data_format() == 'channels_first':
    input_shape = (1, image_width, image_height)
else:
    input_shape = (image_width, image_height, 1)

train_dir = 'dataset/train'                #input
validation_dir = 'dataset/test'            #input
train_samples = 5830                        #input
validation_samples = 5830                    #input

batch_size = 16
epochs = 30

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(8))                             #input
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


train_datagen = ImageDataGenerator(shear_range=0.4, zoom_range=0.0, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(image_width, image_height), batch_size=batch_size, 
                                                    color_mode="grayscale", class_mode='categorical')

test_datagen = ImageDataGenerator()
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(image_width, image_height), batch_size=batch_size, 
                                                        color_mode="grayscale", class_mode='categorical')

model.fit_generator(train_generator, steps_per_epoch= train_samples / batch_size, epochs=epochs, validation_data=validation_generator, 
                    validation_steps= validation_samples / batch_size)

model_json = model.to_json()
with open("model_in_json.json", "w") as json_file:
    json.dump(model_json, json_file)


model.save_weights('model_weights.h5')

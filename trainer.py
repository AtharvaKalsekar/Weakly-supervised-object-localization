from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import accessing_dirs_test
from IDG import DataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, Activation

image_size = 32

model = models.Sequential()

#block-1
model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(32,32,3), padding='SAME'))
model.add(Activation('relu'))

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='SAME'))
model.add(Activation('relu'))

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='SAME'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

#block-2
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='SAME'))
model.add(Activation('relu'))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='SAME'))
model.add(Activation('relu'))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='SAME'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

#block-3
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='SAME'))
model.add(Activation('relu'))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='SAME'))
model.add(Activation('relu'))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='SAME'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

#block-4
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='SAME'))
model.add(Activation('relu'))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='SAME'))
model.add(Activation('relu'))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='SAME'))
model.add(Activation('relu'))

model.add(layers.GlobalAveragePooling2D())

model.add(layers.Dense(10, activation='softmax'))
 
print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=1e-4),
              metrics=['acc'])



partition , labels = get_partition_and_labels()
params={
    'dim': (32,32),
    'patch_size' : 8,
    'batch_size': 16,
    'n_classes': 10,
    'n_channels': 3,
    'shuffle': True,
    
}

train_datagen = DataGenerator(partition['train'], labels, path='cifar/train/', data_purpose='train',**params)
validation_datagen = DataGenerator(partition['validation'], labels, path='cifar/validation/', data_purpose='validation',**params)

model.fit_generator(generator=train_datagen, validation_data=validation_datagen, epochs=30, steps_per_epoch=3125, validation_steps=562, use_multiprocessing=True, workers=6)

model.save('custom_3_16062019.h5')

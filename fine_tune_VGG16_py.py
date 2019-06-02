from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import accessing_dirs_test
from IDG import DataGenerator

image_size = 32
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

for layer in vgg_conv.layers[:-4]:
    layer.trainable = False
 
'''for layer in vgg_conv.layers:
    print(layer, layer.trainable)'''

model = models.Sequential()
 
# Add the vgg convolutional base model
model.add(vgg_conv)
 
# Add new layers
'''model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))'''
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(10, activation='softmax'))
 
'''model.summary()

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)
 
# Change the batchsize according to your system RAM
train_batchsize = 16
val_batchsize = 16

train_dir = "cifar/train"
validation_dir = "cifar/validation"

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=train_batchsize,
        class_mode='categorical')
 
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)'''

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])



partition , labels = accessing_dirs_test.get_partition_and_labels()
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

model.fit_generator(generator=train_datagen, validation_data=validation_datagen, epochs=30, steps_per_epoch=2500, validation_steps=625, use_multiprocessing=True, workers=6)
'''# Train the model
history = model.fit_generator(
      train_datagen,
      steps_per_epoch=train_generator.samples/train_generator.batch_size ,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1)
'''
model.save('fine_tune_10042019.h5')
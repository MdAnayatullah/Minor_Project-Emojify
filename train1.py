import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
from PIL import Image

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input
from keras import Model, layers
from keras.models import load_model, model_from_json

train_datagen = ImageDataGenerator(
    shear_range=10,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    'src/Data/train',
    batch_size=64,
    class_mode='binary',
    target_size=(48,48))

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)

validation_generator = validation_datagen.flow_from_directory(
    'src/Data/test',
    shuffle=False,
    class_mode='binary',
    target_size=(48,48))

conv_base = ResNet50(
    include_top=False,
    weights='imagenet')

for layer in conv_base.layers:
    layer.trainable = False

x = conv_base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x) 
predictions = layers.Dense(2, activation='softmax')(x)
model = Model(conv_base.input, predictions)

optimizer = keras.optimizers.Adam()
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=28709//64,
                              epochs=50,
                              validation_data=validation_generator,
                              validation_steps=7178//64)

# architecture and weights to HDF5
model.save('model.h6')

# architecture to JSON, weights to HDF5
# model.save_weights('weights.h5')
# with open('architecture.json', 'w') as f:
#         f.write(model.to_json())

# # architecture and weights from HDF5
# model = load_model('model.h5')

# # architecture from JSON, weights from HDF5
# with open('architecture.json') as f:
#     model = model_from_json(f.read())
# model.load_weights('weights.h5')
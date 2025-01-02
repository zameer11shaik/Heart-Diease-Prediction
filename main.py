import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3

# Load InceptionV3
inception_v3 = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in inception_v3.layers[:-15]:
    layer.trainable = False

# Add custom layers on top of InceptionV3
x = inception_v3.output
x = Flatten()(x)
x = Dense(units=512, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(units=512, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(units=4, activation='softmax')(x)

model = Model(inception_v3.input, output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
    'Data/train', target_size=(224, 224), batch_size=32, class_mode='categorical')
test_set = train_datagen.flow_from_directory(
    'Data/test', target_size=(224, 224), batch_size=32, class_mode='categorical')

# Train the model
history = model.fit(training_set, validation_data=test_set, epochs=25,
                    steps_per_epoch=len(training_set), validation_steps=len(test_set))

# Save the model
model.save('Models/inception_chest_main2001.h5')

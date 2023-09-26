import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Setzen Sie die Pfade und andere Konfigurationen
train_data_dir = 'Data/train'
valid_data_dir = 'Data/valid'
test_data_dir = 'Data/test'
img_width, img_height = 150, 150
batch_size = 32
num_epochs = 10
num_classes = len(os.listdir(train_data_dir))

# Daten vorverarbeiten und Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    valid_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Erstellen Sie ein Sequenzielles Modell
model = Sequential()

# Fügen Sie Convolutional Layers und Pooling Layers hinzu
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flattening Layer und Fully Connected Layers hinzufügen
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Kompilieren des Modells
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modell trainieren
history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=validation_generator
)

# Das Modell speichern
model.save('bird_classifier_model.h5')

import os
import numpy as np
from PIL import Image
from tensorflow import keras
import random

# Pfade zu den Datenordnern
train_data_dir = 'Data/train/'
valid_data_dir = 'Data/valid/'
bird_image_dir = 'Picture_of_Bird/'  # Pfad zum Ordner mit dem Bild des Vogels
input_image_dir = 'Input_Images/'  # Pfad zum Ordner mit den Eingabebildern

# Laden Sie ein vortrainiertes Modell (z. B. MobileNetV2) oder trainieren Sie ein Modell von Grund auf
# Hier verwenden wir MobileNetV2 als Beispiel
model = keras.applications.MobileNetV2(weights='imagenet', include_top=True)

# Funktion zur Vorhersage des Vogels auf einem Bild
def predict_bird(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Das Modell erwartet 224x224 Pixel Bilder
    img = np.expand_dims(np.array(img), axis=0)
    img = keras.applications.mobilenet_v2.preprocess_input(img)
    predictions = model.predict(img)
    decoded_predictions = keras.applications.mobilenet_v2.decode_predictions(predictions)
    return decoded_predictions[0]

# Funktion zur Überprüfung und Speicherung des Bilds
def process_image(image_path, predicted_class):
    print(f"Das Modell sagt, es ist ein {predicted_class}.")
    user_input = input("Ist das Ergebnis richtig? (Ja/Nein): ").strip().lower()
    if user_input == 'ja':
        bird_folder = os.path.join(train_data_dir, predicted_class)
        os.makedirs(bird_folder, exist_ok=True)
        image = Image.open(image_path)
        image.save(os.path.join(bird_folder, os.path.basename(image_path)))
        print("Bild wurde in den Trainingsordner verschoben.")
    else:
        print("Bild wurde nicht verschoben.")

# Hauptfunktion zum Hochladen und Verarbeiten eines zufälligen Bilds aus "Input_Images"
def main():
    # Liste aller Bilder im Ordner "Input_Images"
    input_images = os.listdir(input_image_dir)
    
    # Zufälliges Bild auswählen
    random_image = random.choice(input_images)
    input_image_path = os.path.join(input_image_dir, random_image)
    
    predicted_class = predict_bird(input_image_path)
    process_image(input_image_path, predicted_class[0][1])

if __name__ == "__main__":
    main()

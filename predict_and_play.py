
import sys
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from emotion_labels import emotion_labels
from music_feature_extractor import MusicGenerator, MusicFeatureExtractor

def load_and_preprocess_image(image_path, target_size=(48, 48)):
    img = Image.open(image_path).convert('L')
    img = img.resize(target_size)
    arr = np.array(img).astype('float32') / 255.0
    return arr.reshape((1, 48, 48, 1))

def predict_emotion(image_path, model_path):
    model = tf.keras.models.load_model(model_path)
    inp = load_and_preprocess_image(image_path)
    preds = model.predict(inp)
    idx = int(np.argmax(preds))
    return emotion_labels[idx]

def main(image_path):
    model_path = 'final_model.keras'
    emotion = predict_emotion(image_path, model_path)
    print(f"Predicted emotion: {emotion}")
    extractor = MusicFeatureExtractor()
    generator = MusicGenerator(features_dict=extractor.emotion_features)
    generator.generate(emotion, duration=30, output_file=f"{emotion}_generated.mid")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict_and_play.py <image_path>")
    else:
        main(sys.argv[1])

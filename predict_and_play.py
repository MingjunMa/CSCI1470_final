import os
import random
import numpy as np
import tensorflow as tf
from PIL import Image
import pygame
from emotion_labels import emotion_labels

def load_and_preprocess_image(image_path, target_size=(48, 48)):
    img = Image.open(image_path).convert('L')  # 转为灰度图
    img = img.resize(target_size)
    img_array = np.array(img).astype('float32') / 255.0
    img_array = img_array.reshape((1, 48, 48, 1))
    return img_array

def predict_emotion(image_path, model_path='/Users/mamingjun/Desktop/1470/final_project/final_model.keras'):
    model = tf.keras.models.load_model(model_path)
    img_array = load_and_preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_emotion = emotion_labels[predicted_index]
    confidence = np.max(prediction)
    print(f"\n😃 Predicted Emotion: {predicted_emotion} (Confidence: {confidence:.2f})")
    return predicted_emotion

def play_music(emotion, music_base='/Users/mamingjun/Desktop/1470/final_project/data/emotion_music_data'):
    emotion_dir = os.path.join(music_base, emotion)
    if not os.path.exists(emotion_dir):
        print(f"[⚠️] No music found for emotion: {emotion}")
        return

    songs = [f for f in os.listdir(emotion_dir) if f.endswith('.mp3') or f.endswith('.wav')]
    if not songs:
        print(f"[⚠️] No audio files in: {emotion_dir}")
        return

    song_path = os.path.join(emotion_dir, random.choice(songs))

    print(f"🎵 Now playing: {song_path}")
    pygame.mixer.init()
    pygame.mixer.music.load(song_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        continue

def main(image_path):
    model_path = '/Users/mamingjun/Desktop/1470/final_project/final_model.keras'
    predicted_emotion = predict_emotion(image_path, model_path=model_path)
    play_music(predicted_emotion)

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict_and_play.py <image_path>")
    else:
        main(sys.argv[1])
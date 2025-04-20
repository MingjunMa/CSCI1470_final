import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def load_test_dataset(data_dir='data/FER2013', image_size=(48, 48), batch_size=64):
    test_dir = os.path.join(data_dir, 'test')
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='categorical',
        image_size=image_size,
        color_mode='grayscale',
        batch_size=batch_size,
        shuffle=False
    )

    def normalize(image, label):
        return tf.cast(image, tf.float32) / 255.0, label

    test_ds = test_ds.map(normalize).prefetch(tf.data.AUTOTUNE)
    return test_ds

def evaluate_model(model_path='/Users/mamingjun/Desktop/1470/final_project/best_model.keras', data_dir='/Users/mamingjun/Desktop/1470/final_project/data/FER2013'):
    model = tf.keras.models.load_model(model_path)
    test_ds = load_test_dataset(data_dir=data_dir)

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=emotion_labels))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=emotion_labels, yticklabels=emotion_labels, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_model()

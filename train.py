import os
import tensorflow as tf
from data_loader import load_fer2013_from_folders
from model import build_emotion_model

def train_emotion_model(
    data_dir='/Users/mamingjun/Desktop/1470/final_project/data/FER2013',
    save_dir='/Users/mamingjun/Desktop/1470/final_project',
    image_size=(48, 48),
    batch_size=64,
    num_classes=7,
    epochs=30
):

    train_ds, val_ds = load_fer2013_from_folders(data_dir=data_dir, image_size=image_size, batch_size=batch_size)

    model = build_emotion_model(input_shape=(image_size[0], image_size[1], 1), num_classes=num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    os.makedirs(save_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_dir, 'best_model.keras'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        )
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    model.save(os.path.join(save_dir, 'final_model.keras'))
    print("✅ Training complete! Model saved to:", save_dir)

if __name__ == "__main__":
    train_emotion_model()
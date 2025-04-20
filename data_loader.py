import tensorflow as tf
import os

def load_fer2013_from_folders(data_dir='data/FER2013', image_size=(48, 48), batch_size=64):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='categorical',
        image_size=image_size,
        color_mode='grayscale',
        batch_size=batch_size,
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
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

    train_ds = train_ds.map(normalize).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(normalize).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds
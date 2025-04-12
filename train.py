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
    # 加载数据
    train_ds, val_ds = load_fer2013_from_folders(data_dir=data_dir, image_size=image_size, batch_size=batch_size)

    # 构建模型
    model = build_emotion_model(input_shape=(image_size[0], image_size[1], 1), num_classes=num_classes)

    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 创建保存路径
    os.makedirs(save_dir, exist_ok=True)

    # 设置回调
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_dir, 'best_model.keras'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        )
    ]

    # 训练模型
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # 保存最终模型
    model.save(os.path.join(save_dir, 'final_model.keras'))
    print("✅ Training complete! Model saved to:", save_dir)

if __name__ == "__main__":
    train_emotion_model()
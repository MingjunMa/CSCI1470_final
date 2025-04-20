# CSCI1470_final
final project for CSCI1470


Our project is a deep learning-based system that recognizes facial emotions from user-provided images and recommends music that aligns with the detected affective state. It integrates computer vision and emotion-aware music curation in a simple, multimodal pipeline.

This project connects emotion recognition and music generation through:

- **Facial Expression Recognition** using a CNN trained from scratch on the [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- **Emotion Label Mapping** using Valence-Arousal scores from the [DEAM music dataset](https://cvml.unige.ch/databases/DEAM/)

- **data_loader.py**         # FER2013 image loader
- **model.py**               # CNN model definition
- **train.py**               # Train emotion classifier from scratch
- **test.py**                # Evaluate on test set
- **organize_deam_music.py**     # Convert DEAM valence/arousal to 7-class emotions
- **music_feature_extractor.py** # extract important features from 7-class emotions
- **predict_and_play.py**    # Main script: image → emotion → music

DATA:
Since the dataset was too large, we uploaded it to Google Drive: https://drive.google.com/file/d/17dvkc6lA6_JHmCUV8P_-Bhld3FDVZGaL/view?usp=sharing

RUN:
If you want to train our model again, you can use:
python train.py

If you want to run motion-to-music demo, you can use:
python predict_and_play.py test_face.jpg

import os
import shutil
import pandas as pd

import random

def map_valence_arousal_to_emotion(valence, arousal):

    if valence >= 6 and arousal >= 6:
        return random.choice(['happy', 'surprise'])
    elif valence >= 6 and arousal <= 4:
        return 'calm'
    elif valence <= 4 and arousal >= 6:
        return random.choice(['angry', 'fear'])
    elif valence <= 4 and arousal <= 4:
        return random.choice(['sad', 'disgust'])
    else:
        return 'neutral'

def organize_deam_by_emotion(
    csv_path_1='/Users/mamingjun/Desktop/1470/final_project/data/static_annotations_averaged_songs_1_2000.csv',
    csv_path_2='/Users/mamingjun/Desktop/1470/final_project/data/static_annotations_averaged_songs_2000_2058.csv',
    music_path='/Users/mamingjun/Desktop/1470/final_project/data/MEMD_audio',
    output_dir='/Users/mamingjun/Desktop/1470/final_project/data/emotion_music_data'
):
    df1 = pd.read_csv(csv_path_1)
    df2 = pd.read_csv(csv_path_2)
    df = pd.concat([df1, df2], ignore_index=True)

    count = 0
    for _, row in df.iterrows():
        song_id = f"{int(row['song_id'])}.mp3" 
        valence = row[' valence_mean']
        arousal = row[' arousal_mean']
        emotion = map_valence_arousal_to_emotion(valence, arousal)

        src_path = os.path.join(music_path, song_id)
        dst_dir = os.path.join(output_dir, emotion)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, song_id)

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            count += 1
        else:
            print(f"Missing file: {src_path}")

    print(f"\n Copied {count} songs into {output_dir}/[emotion]/ folders.")

if __name__ == "__main__":
    organize_deam_by_emotion()
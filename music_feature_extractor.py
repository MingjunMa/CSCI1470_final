
import os
import json
import numpy as np
import librosa
import random
from collections import defaultdict
from mido import MidiFile, MidiTrack, Message, MetaMessage

NOTE_TO_MIDI = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
}

CHORD_TO_NOTES = {
    'I': ['C', 'E', 'G'], 'ii': ['D', 'F', 'A'], 'iii': ['E', 'G', 'B'],
    'IV': ['F', 'A', 'C'], 'V': ['G', 'B', 'D'], 'vi': ['A', 'C', 'E'], 'vii°': ['B', 'D', 'F'],
    'i': ['C', 'Eb', 'G'], 'ii°': ['D', 'F', 'Ab'], 'III': ['Eb', 'G', 'Bb'],
    'iv': ['F', 'Ab', 'C'], 'v': ['G', 'Bb', 'D'], 'VI': ['Ab', 'C', 'Eb'], 'VII': ['Bb', 'D', 'F'],
    'bVII': ['Bb', 'D', 'F']
}

class MusicFeatureExtractor:
    def __init__(self, data_dir='data/emotion_music_data', cache_file='data/emotion_music_features.json'):
        self.data_dir = data_dir
        self.cache_file = cache_file
        self.emotion_features = {}
        self.load_or_extract_features()

    def load_or_extract_features(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                self.emotion_features = json.load(f)
        else:
            self.extract_features()
            self.save_features()
            
    def extract_features(self):
        print("Extracting features from music files...")
        for emotion in os.listdir(self.data_dir):
            emotion_dir = os.path.join(self.data_dir, emotion)
            if not os.path.isdir(emotion_dir):
                continue
                
            print(f"Processing {emotion} music files...")
            features = self.analyze_emotion_music(emotion_dir)
            self.emotion_features[emotion] = features
    
    def analyze_emotion_music(self, emotion_dir):
        tempos = []
        pitches = []
        velocities = []
        note_densities = []
        modes = []
        chord_progressions = []
        
        music_files = [os.path.join(emotion_dir, f) for f in os.listdir(emotion_dir) 
                      if f.endswith('.mp3') or f.endswith('.wav')]
        
        sample_size = min(10, len(music_files))
        sampled_files = random.sample(music_files, sample_size) if len(music_files) > sample_size else music_files
        
        for music_file in sampled_files:
            try:
                y, sr = librosa.load(music_file, sr=None)
                
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                tempos.append(tempo)
                
                pitches = librosa.feature.chroma_cqt(y=y, sr=sr)
                pitch_mean = np.mean(pitches) * 127  
                pitch_std = np.std(pitches) * 127
                pitch_min = max(0, int(pitch_mean - pitch_std))  
                pitch_max = min(127, int(pitch_mean + pitch_std))
                
                rms = librosa.feature.rms(y=y)[0]
                velocity_mean = np.mean(rms) * 127  
                velocity_std = np.std(rms) * 127
                velocity_min = max(40, int(velocity_mean - velocity_std))  
                velocity_max = min(127, int(velocity_mean + velocity_std))
                
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                density = np.mean(onset_env)
                if density < 0.1:
                    note_densities.append("low")
                elif density < 0.2:
                    note_densities.append("medium")
                else:
                    note_densities.append("high")
                
                chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
                chroma_sum = np.sum(chroma, axis=1)
                major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])  
                minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])  
                
                major_corr = np.corrcoef(chroma_sum, major_profile)[0, 1]
                minor_corr = np.corrcoef(chroma_sum, minor_profile)[0, 1]
                
                if major_corr > minor_corr:
                    modes.append("major")
                else:
                    modes.append("minor")
                    
                if "major" in modes:
                    chord_progressions.append(["I", "IV", "V", "I"])
                else:
                    chord_progressions.append(["i", "VI", "III", "VII"])
                    
            except Exception as e:
                print(f"Error processing {music_file}: {e}")
                continue
        
        if not tempos:  
            return self.get_default_features()
            
        tempo_min = max(60, int(np.percentile(tempos, 25)))
        tempo_max = min(180, int(np.percentile(tempos, 75)))
        
        mode = max(set(modes), key=modes.count) if modes else "major"
        
        density = max(set(note_densities), key=note_densities.count) if note_densities else "medium"
        
        if mode == "major":
            chord_prog = ["I", "IV", "V", "I"]
        else:
            chord_prog = ["i", "VI", "III", "VII"]
        
        rhythm_pattern = "regular"
        if "angry" in emotion_dir.lower() or "fear" in emotion_dir.lower() or "surprise" in emotion_dir.lower():
            rhythm_pattern = "irregular"
        
        return {
            "tempo": [tempo_min, tempo_max],
            "mode": mode,
            "pitch_range": [60, 84],  
            "velocity_range": [60, 100],  
            "note_density": density,
            "rhythm_pattern": rhythm_pattern,
            "chord_progression": chord_prog
        }
    
    def get_default_features(self):
        return {
            "tempo": [90, 120],
            "mode": "major",
            "pitch_range": [60, 84],
            "velocity_range": [60, 100],
            "note_density": "medium",
            "rhythm_pattern": "regular",
            "chord_progression": ["I", "IV", "V", "I"]
        }
        
    def save_features(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.emotion_features, f, indent=2)

class MusicGenerator:
    def __init__(self, features_dict):
        self.features = features_dict

    def generate(self, emotion, duration=30, output_file='generated_music.mid'):
        if emotion not in self.features:
            raise ValueError(f"Emotion '{emotion}' not found in features.")
        f = self.features[emotion]
        tempo = int(np.random.uniform(f['tempo'][0], f['tempo'][1]))
        pitch_range = f['pitch_range']
        velocity_range = f['velocity_range']
        chord_prog = f['chord_progression']
        density = f['note_density']
        mode = f['mode']
        rhythm = f['rhythm_pattern']

        mid = MidiFile(type=1)
        ticks_per_beat = 480
        mid.ticks_per_beat = ticks_per_beat
        beats_per_bar = 4
        ticks_per_bar = beats_per_bar * ticks_per_beat

        tempo_track = MidiTrack()
        mid.tracks.append(tempo_track)
        tempo_us = int(60000000 / tempo)
        tempo_track.append(MetaMessage('set_tempo', tempo=tempo_us, time=0))

        chord_track = MidiTrack()
        melody_track = MidiTrack()
        mid.tracks.append(chord_track)
        mid.tracks.append(melody_track)
        chord_track.append(Message('program_change', program=0, time=0))
        melody_track.append(Message('program_change', program=0, time=0))

        if density == 'low':
            notes_per_bar = 4
        elif density == 'medium':
            notes_per_bar = 8
        else:
            notes_per_bar = 16
        note_duration = ticks_per_bar // notes_per_bar
        bars = int(duration * tempo / 60 / beats_per_bar)

        current_time = 0
        for i in range(bars):
            chord = chord_prog[i % len(chord_prog)]
            chord_notes = CHORD_TO_NOTES[chord]
            chord_midi = [np.clip(NOTE_TO_MIDI[n] + 60, 0, 127) for n in chord_notes]

            # chords
            for idx, note in enumerate(chord_midi):
                chord_track.append(Message('note_on', note=note, velocity=80, time=current_time if idx == 0 else 0))
            for idx, note in enumerate(chord_midi):
                chord_track.append(Message('note_off', note=note, velocity=0, time=ticks_per_bar if idx == 0 else 0))

            # melody
            for _ in range(notes_per_bar):
                pitch = np.random.randint(pitch_range[0], pitch_range[1]+1)
                velocity = np.random.randint(velocity_range[0], velocity_range[1]+1)
                pitch = int(np.clip(pitch, 0, 127))
                velocity = int(np.clip(velocity, 0, 127))

                note_ticks = note_duration
                if rhythm == "irregular":
                    note_ticks = int(note_duration * np.random.choice([0.5, 1.0, 1.5], p=[0.3, 0.5, 0.2]))
                    note_ticks = max(60, note_ticks)  # 防止太短
                
                melody_track.append(Message('note_on', note=pitch, velocity=velocity, time=0))
                melody_track.append(Message('note_off', note=pitch, velocity=0, time=note_ticks))

            current_time = 0

        os.makedirs('output', exist_ok=True)
        full_path = os.path.join('output', output_file)
        mid.save(full_path)
        return full_path
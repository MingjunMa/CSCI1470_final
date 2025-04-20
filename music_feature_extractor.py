
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
        ticks_per_bar = ticks_per_beat * beats_per_bar

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
            chord_midi = [NOTE_TO_MIDI[n] + 60 for n in chord_notes]

            for note in chord_midi:
                chord_track.append(Message('note_on', note=note, velocity=80, time=current_time))
            for note in chord_midi:
                chord_track.append(Message('note_off', note=note, velocity=0, time=ticks_per_bar if note == chord_midi[0] else 0))

            for _ in range(notes_per_bar):
                pitch = np.random.randint(pitch_range[0], pitch_range[1]+1)
                pitch = np.clip(pitch, 0, 127)
                velocity = np.random.randint(velocity_range[0], velocity_range[1]+1)
                velocity = int(np.clip(velocity, 0, 127))
                melody_track.append(Message('note_on', note=pitch, velocity=velocity, time=0))
                melody_track.append(Message('note_off', note=pitch, velocity=0, time=note_duration))

        os.makedirs('output', exist_ok=True)
        full_path = os.path.join('output', output_file)
        mid.save(full_path)
        return full_path

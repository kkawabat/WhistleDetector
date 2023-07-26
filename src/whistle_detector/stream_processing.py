import threading
import time
from collections import Counter

import numpy as np
import pyaudio

from kaudio.mic_engine import MicEngine
import matplotlib.pyplot as plt
import sounddevice as sd
from src.whistle_detector.audio_processing import midi_note_to_key, notes_to_midi, whistle_to_midi_notes_frame


def get_note(in_data, frame_count, time_info, status_flags):
    global i, audio_frame_mtx, NOTE_tracker_list
    lapse_time = time.time() - start_time
    audio_data = np.frombuffer(in_data, dtype=np.int16)
    audio_frame_mtx[:, i] = audio_data
    i += 1
    duration, midi_notes, volume = whistle_to_midi_notes_frame(SAMPLING_RATE, audio_data)
    if volume > 30:
        NOTE_tracker_list.append([lapse_time, duration, midi_notes, volume])

    freq_octave, freq_note = midi_note_to_key([midi_notes])
    mode_note, count = Counter([f'{note}{octave}' for note, octave in zip(freq_note, freq_octave)]).most_common(1)[0]
    print(mode_note, count, volume)
    if i == 50:
        t, duration, midi_notes, volume = list(zip(*NOTE_tracker_list))

        print(t)
        print(duration)
        print(midi_notes)
        print(volume)
        notes_to_midi(t, duration, midi_notes, volume, 'output.mid')
        ME.stop_stream()

    return audio_data, pyaudio.paContinue


if __name__ == '__main__':
    start_time = time.time()
    SAMPLING_RATE = 44100
    BUFFER_SIZE = 2 ** 14
    ME = MicEngine()
    ME.init_stream(SAMPLING_RATE, BUFFER_SIZE, get_note)
    audio_frame_mtx = np.empty((BUFFER_SIZE, 100))
    NOTE_tracker_list = []
    i = 0

    try:
        ME.start_stream()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ME.stop_stream()
    pass
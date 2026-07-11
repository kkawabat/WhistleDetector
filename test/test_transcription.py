"""Regression tests for note transcription. Run from the repo root:

    python -m pytest test/test_transcription.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from whistle_detector.audio_processing import WhistleDetector

FS = 16000
TEST_DATA = os.path.join(os.path.dirname(__file__), 'test_data')


def tone(freq, dur, amp_env=None):
    t = np.arange(int(FS * dur)) / FS
    env = np.minimum(1, np.minimum(t / 0.02, (dur - t) / 0.02))
    if amp_env is not None:
        env = env * amp_env(t)
    return 8000 * env * np.sin(2 * np.pi * freq * t)


def pad(seg):
    return np.concatenate([np.zeros(FS // 4), seg, np.zeros(FS // 4)]).astype(np.int16)


def notes(audio):
    wd = WhistleDetector.from_audio(pad(audio), FS)
    return [f"{k}{o}" for k, o in zip(wd.key, wd.key_octave)]


def test_single_note():
    assert notes(tone(587.33, 0.5)) == ['D5']


def test_breath_flutter_is_one_note():
    flutter = lambda t: 1 - 0.2 * np.exp(-((t - 0.4) / 0.05) ** 2)
    assert notes(tone(880, 0.8, flutter)) == ['A5']


def test_double_tongued_is_two_notes():
    seg = np.concatenate([tone(880, 0.35), np.zeros(int(0.1 * FS)), tone(880, 0.35)])
    assert notes(seg) == ['A5', 'A5']


def test_onset_scoop_is_one_note():
    t = np.arange(int(FS * 0.6)) / FS
    freq = np.where(t < 0.1, 880 + (987.77 - 880) * (t / 0.1), 987.77)
    env = np.minimum(1, np.minimum(t / 0.02, (0.6 - t) / 0.02))
    assert notes(8000 * env * np.sin(2 * np.pi * np.cumsum(freq) / FS)) == ['B5']


def test_chromatic_legato_preserved():
    seg = np.concatenate([tone(880, 0.3)[:-320], tone(932.33, 0.3)])
    assert notes(seg) == ['A5', 'A#5']


def test_legato_arpeggio_splits():
    seg = np.concatenate([tone(523.25, 0.4)[:-320], tone(659.25, 0.4)[:-320], tone(783.99, 0.4)])
    assert notes(seg) == ['C5', 'E5', 'G5']


def test_staccato_short_notes_survive():
    seg = np.concatenate([tone(880, 0.08), np.zeros(int(0.15 * FS)), tone(987.77, 0.08)])
    assert notes(seg) == ['A5', 'B5']


def test_chromatic_scale_pitch_accuracy():
    freqs = 440 * 2 ** (np.arange(3, 15) / 12)
    seg = np.concatenate([np.concatenate([tone(f, 0.25), np.zeros(int(0.1 * FS))]) for f in freqs])
    assert notes(seg) == ['C5', 'C#5', 'D5', 'D#5', 'E5', 'F5', 'F#5',
                          'G5', 'G#5', 'A5', 'A#5', 'B5']


def test_vibrato_is_one_note():
    t = np.arange(int(FS * 1.0)) / FS
    phase = 2 * np.pi * 880 * t + (880 * 0.035 / 5) * np.sin(2 * np.pi * 5 * t)
    env = np.minimum(1, np.minimum(t / 0.02, (1.0 - t) / 0.02))
    assert notes(8000 * env * np.sin(phase)) == ['A5']


def test_twinkle_recording_note_count():
    # real browser recording of the first 7 notes of twinkle twinkle little
    # star; quiet notes fade under the noise floor mid-note and every note
    # has onset/release pitch transients
    wd = WhistleDetector.from_webm(os.path.join(TEST_DATA, 'whistle_sample_twinkle.weba'))
    assert len(wd.midi_notes) == 7
    # repeated blows of the same pitch must transcribe as repeats
    assert wd.key[0] == wd.key[1]  # C C
    assert wd.key[4] == wd.key[5]  # A A

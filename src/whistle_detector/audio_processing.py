import base64
import io
import tempfile

import ffmpeg
import numpy as np
from midiutil import MIDIFile
from scipy import stats
from scipy.io import wavfile
from scipy.signal import butter, lfilter, savgol_filter, find_peaks, spectrogram

MIN_WHISTLE_DURATION = .05
MAX_MIDI_VOLUMN = 127
MAX_WHISTLE_FREQ = 5000
MIN_WHISTLE_FREQ = 500

# A4 = 440 Hz = 58 pitch index for midi
REFERENCE_FREQUENCY = 440
REFERENCE_MIDI_PITCH = 58

NOTES = np.array(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])


class WhistleDetector:
    def __init__(self, time, duration, midi_notes, volume, key_octave, key):
        self.time = time
        self.duration = duration
        self.midi_notes = midi_notes
        self.volume = volume
        self.key_octave = key_octave
        self.key = key

    @classmethod
    def from_audio(cls, audio, fs):
        time_vec, pitch_vec, amp_vec = cls.audio_to_pitch(fs, audio)
        time, duration, midi_notes, volume = cls.pitch_to_notes(time_vec, pitch_vec, amp_vec)
        key_octave, key = cls.midi_note_to_key(midi_notes)
        return cls(time, duration, midi_notes, volume, key_octave, key)

    @classmethod
    def from_wav(cls, wav_path):
        fs, audio = wavfile.read(wav_path)
        return cls.from_audio(audio, fs)

    @classmethod
    def from_webm(cls, webm_path):
        fs, audio = cls.webm_to_audio(webm_path)
        return cls.from_audio(audio, fs)

    @classmethod
    def from_webm_blob(cls, webm_blob):
        fs, audio = cls.webm_blob_to_audio(webm_blob)
        return cls.from_audio(audio, fs)

    @staticmethod
    def webm_to_audio(audio_blob, fs=16000):
        try:
            out, err = (
                ffmpeg
                .input(audio_blob)
                .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar=str(fs))
                .run(cmd='ffmpeg', capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            print('stdout:', e.stdout.decode('utf8'))
            print('stderr:', e.stderr.decode('utf8'))
            raise e

        return fs, np.frombuffer(out, np.int16)

    @classmethod
    def webm_blob_to_audio(cls, webm_blob, fs=16000):
        with tempfile.TemporaryFile(suffix='.weba', delete=False) as f:
            f.write(webm_blob.read())
            f.seek(0)
            fs, audio = cls.webm_to_audio(f.name, fs=fs)
        return fs, audio

    @classmethod
    def pitch_to_notes(cls, time_vec, pitch_vec, amp_vec):
        raw_midi_notes = cls._pitch_to_notes(pitch_vec)
        time, duration, midi_notes, volume = cls._smooth_notes_by_volume(raw_midi_notes, time_vec, amp_vec)
        return time, duration, midi_notes, volume

    @classmethod
    def _pitch_to_notes(cls, pitch_vec):
        # algorithm taken from https://stackoverflow.com/a/64505498
        # note index = log_o(freq/440) where o = 2^(1/12) the diff of frequency between consecutive notes
        note_multiplier = 2 ** (1 / 12)
        distance_from_known_note = np.round(np.log(pitch_vec / REFERENCE_FREQUENCY) / np.log(note_multiplier))
        midi_notes = REFERENCE_MIDI_PITCH + distance_from_known_note

        # if midi note is out of range it's likely a bad frame, set value to nan
        midi_notes[midi_notes < 0] = float('nan')
        return midi_notes

    @staticmethod
    def _smooth_notes_by_volume(raw_midi_notes, time_vec, amp_vec):
        peak_idx, peak_property = find_peaks(amp_vec, width=MIN_WHISTLE_DURATION / time_vec[0], prominence=100)
        peak_width, peak_prominence = peak_property['widths'], peak_property['prominences']
        if len(peak_idx) == 0:
            return [], [], [], []

        duration = peak_width * time_vec[0]
        time = time_vec[peak_idx]
        volume = np.round(MAX_MIDI_VOLUMN * (np.log(peak_prominence) / np.log(max(peak_prominence)))).astype(int)

        midi_notes = []
        for i, w in zip(peak_idx, peak_width):
            hw = int(w / 2)
            # for duration of the whistle find the mode note
            mode, mode_count = stats.mode(raw_midi_notes[max((i - hw), 0): min((i + hw), len(time_vec))])
            midi_notes.append(int(mode))

        return time, duration, midi_notes, volume

    @classmethod
    def audio_to_pitch(cls, sampling_rate, audio_vec, nfft=256):
        audio_vec = cls._butter_bandpass_filter(audio_vec, MIN_WHISTLE_FREQ, MAX_WHISTLE_FREQ, sampling_rate)
        freq_vec, time_vec, spec_mtx = spectrogram(audio_vec, fs=sampling_rate, nfft=nfft)
        pitch = freq_vec[spec_mtx.argmax(axis=0)]
        win_length = min(51, len(time_vec) if len(time_vec) % 2 == 1 else len(time_vec) - 1)
        amp_vec = savgol_filter(spec_mtx.max(axis=0), win_length, 3)
        amp_vec[amp_vec < 1] = 1
        return time_vec, pitch, amp_vec

    @staticmethod
    def _butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, data)
        return y

    @staticmethod
    def midi_note_to_key(midi_notes):
        if len(midi_notes) == 0:
            return np.array([]), np.array([])

        key_octave = [num // len(NOTES) for num in midi_notes]
        key = NOTES[[int(num % len(NOTES)) for num in midi_notes]]
        return key_octave, key

    def construct_midi_file(self):
        track, channel, tempo = 0, 0, 60
        mfile = MIDIFile(1)  # One track, defaults to format 1 (tempo track automatically created)
        mfile.addTempo(track, 0, tempo)
        for t, d, n, v in zip(self.time, self.duration, self.midi_notes, self.volume):
            mfile.addNote(track, channel, n, t, d, v)
        return mfile

    def to_midi_file(self, output_path):
        mfile = self.construct_midi_file()
        with open(output_path, "wb") as output_file:
            mfile.writeFile(output_file)

    def to_midi_blob(self):
        mfile = self.construct_midi_file()
        with io.BytesIO() as output_file:
            mfile.writeFile(output_file)
            output_file.seek(0)
            blob = base64.b64encode(output_file.read())
        return blob


if __name__ == '__main__':
    sample_audio_path = '../../test/test_data/whistle_sample_stairwaytoheaven2.wav'
    sample_midi_output_path = '../../test/test_output/whistle_sample_stairwaytoheaven2.mid'
    WhistleDetector.from_wav(sample_audio_path).to_midi_file(sample_midi_output_path)
    # WhistleDetector.from_webm()

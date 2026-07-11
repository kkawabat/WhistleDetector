import base64
import io
import os
import tempfile

import ffmpeg
import numpy as np
from midiutil import MIDIFile
from scipy import stats
from scipy.io import wavfile
from scipy.signal import butter, lfilter, savgol_filter, find_peaks, spectrogram

MIN_WHISTLE_DURATION = .05
# volume-envelope smoothing: long enough to suppress frame noise, short
# enough that staccato notes remain separate amplitude peaks
AMP_SMOOTHING_DURATION = .15
MIN_PEAK_PROMINENCE = 100
# a note extends over the contiguous region above this fraction of its peak
# power; envelope peaks whose extents overlap are one blow, not two notes
NOTE_EXTENT_FLOOR = .05
# a segment this much quieter than the note before it (in power), resuming at
# about the same pitch within MAX_RELEASE_GAP seconds, is that note's release
# tail rather than a new note
RELEASE_QUIET_RATIO = .25
MAX_RELEASE_GAP = .15
# a pitch run bordering a one-semitone neighbor must be held this long to be
# a deliberate note rather than onset scoop/drift
MIN_STABLE_NOTE_DURATION = .2
# a wide (3 semitone) onset/release transient must be shorter than this
MAX_SCOOP_DURATION = .15
MAX_MIDI_VOLUMN = 127
MAX_WHISTLE_FREQ = 5000
MIN_WHISTLE_FREQ = 500

# A4 = 440 Hz = MIDI note 69
REFERENCE_FREQUENCY = 440
REFERENCE_MIDI_PITCH = 69

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
    def from_webm_blob(cls, webm_blob):
        fs, audio = cls.webm_blob_to_audio(webm_blob)
        return cls.from_audio(audio, fs)

    @classmethod
    def webm_blob_to_audio(cls, webm_blob, fs=16000):
        with tempfile.NamedTemporaryFile(suffix='.weba', delete=False) as f:
            f.write(webm_blob.read())
            f.seek(0)
            fs, audio = cls.webm_to_audio(f.name, fs=fs)
            f.close()
            os.unlink(f.name)  # required for windows https://stackoverflow.com/questions/23212435/permission-denied-to-write-to-my-temporary-file
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
        with np.errstate(divide='ignore'):  # silent frames detect as 0 Hz
            distance_from_known_note = np.round(np.log(pitch_vec / REFERENCE_FREQUENCY) / np.log(note_multiplier))
        midi_notes = REFERENCE_MIDI_PITCH + distance_from_known_note

        # if midi note is out of range it's likely a bad frame, set value to nan
        midi_notes[midi_notes < 0] = float('nan')
        return midi_notes

    @classmethod
    def _smooth_notes_by_volume(cls, raw_midi_notes, time_vec, amp_vec):
        # time_vec entries are frame-center times; the frame period is their spacing
        dt = time_vec[1] - time_vec[0] if len(time_vec) > 1 else MIN_WHISTLE_DURATION
        min_frames = max(1, int(round(MIN_WHISTLE_DURATION / dt)))
        peak_idx, peak_property = find_peaks(amp_vec, width=min_frames, prominence=MIN_PEAK_PROMINENCE)
        if len(peak_idx) == 0:
            return [], [], [], []

        # frames quieter than the minimum peak prominence are gaps/noise, not notes
        raw_midi_notes = np.where(amp_vec < MIN_PEAK_PROMINENCE, float('nan'), raw_midi_notes)

        # a note's extent is the contiguous region above a small fraction of
        # its peak: half-prominence widths clipped notes to a fraction of
        # their real length, and one blow can flutter in volume, registering
        # several envelope peaks whose extents overlap — those are one note
        segments = []
        for peak, prom in zip(peak_idx, peak_property['prominences']):
            floor = max(MIN_PEAK_PROMINENCE, NOTE_EXTENT_FLOOR * amp_vec[peak])
            start = peak
            while start > 0 and amp_vec[start - 1] >= floor:
                start -= 1
            end = peak
            while end < len(amp_vec) - 1 and amp_vec[end + 1] >= floor:
                end += 1
            if segments and start <= segments[-1]['end']:
                segments[-1]['end'] = max(segments[-1]['end'], end)
                segments[-1]['height'] = max(segments[-1]['height'], amp_vec[peak])
                segments[-1]['prom'] = max(segments[-1]['prom'], prom)
            else:
                segments.append({'start': start, 'end': end, 'height': amp_vec[peak], 'prom': prom})

        max_prom = max(seg['prom'] for seg in segments)
        stable_frames = max(min_frames, int(round(MIN_STABLE_NOTE_DURATION / dt)))
        scoop_frames = max(min_frames, int(round(MAX_SCOOP_DURATION / dt)))

        time, duration, midi_notes, volume = [], [], [], []
        prev_seg = None
        for seg in segments:
            # a sustained whistle is one envelope segment but may contain several
            # notes played legato; split it wherever the pitch settles on a new note
            runs = cls._pitch_runs(raw_midi_notes[seg['start']:seg['end'] + 1], min_frames, stable_frames, scoop_frames)
            if not runs:
                continue

            # a much quieter fragment right after a note, resuming at (about)
            # the same pitch, is the note's release tail — the whistle fading
            # under the noise floor and briefly back — not a new note
            if (prev_seg is not None and midi_notes
                    and len(runs) == 1
                    and (seg['start'] - prev_seg['end']) * dt < MAX_RELEASE_GAP
                    and seg['height'] < RELEASE_QUIET_RATIO * prev_seg['height']
                    and abs(runs[0][2] - midi_notes[-1]) <= 1):
                duration[-1] = time_vec[seg['end']] + dt - time[-1]
                prev_seg = {**prev_seg, 'end': seg['end']}
                continue

            vol = int(np.round(MAX_MIDI_VOLUMN * (np.log(seg['prom']) / np.log(max_prom))))
            for run_start, run_frames, note in runs:
                time.append(time_vec[seg['start'] + run_start])
                duration.append(run_frames * dt)
                midi_notes.append(int(note))
                volume.append(vol)
            prev_seg = seg

        return time, duration, midi_notes, volume

    @classmethod
    def _pitch_runs(cls, window_notes, min_frames, stable_frames, scoop_frames):
        """Split a segment's per-frame notes into (start, length, note) runs of
        stable pitch, dropping runs shorter than min_frames."""
        smoothed = cls._nan_median_smooth(window_notes)
        runs = []
        run_start = None
        for j in range(len(smoothed) + 1):
            val = smoothed[j] if j < len(smoothed) else float('nan')
            if run_start is not None and (np.isnan(val) or val != smoothed[run_start]):
                runs.append((run_start, j - run_start, smoothed[run_start]))
                run_start = None
            if run_start is None and j < len(smoothed) and not np.isnan(val):
                run_start = j

        long_runs = [r for r in runs if r[1] >= min_frames]
        if long_runs:
            merged = [long_runs[0]]
            for start, length, note in long_runs[1:]:
                prev_start, prev_length, prev_note = merged[-1]
                # vibrato can wobble across a semitone boundary and split one
                # held note; re-join same-note runs separated by a short gap
                if note == prev_note and start - (prev_start + prev_length) < min_frames:
                    merged[-1] = (prev_start, start + length - prev_start, note)
                # a one-semitone neighbor that isn't held long enough on one
                # side is onset scoop/drift within a single note, not legato:
                # absorb it into whichever run dominates
                elif abs(note - prev_note) == 1 and min(length, prev_length) < stable_frames:
                    dominant = prev_note if prev_length >= length else note
                    merged[-1] = (prev_start, start + length - prev_start, dominant)
                else:
                    merged.append((start, length, note))

            # the whistle's pitch scoops at onset and sags/rises at release; a
            # brief edge run near its neighbor's pitch is that transient, not
            # a separate note — the wider the interval, the shorter the run
            # must be to count as a transient
            def is_edge_scoop(edge, neighbor):
                if edge[1] >= neighbor[1]:
                    return False
                interval = abs(edge[2] - neighbor[2])
                return ((interval <= 2 and edge[1] < stable_frames)
                        or (interval == 3 and edge[1] < scoop_frames))

            if len(merged) >= 2 and is_edge_scoop(merged[0], merged[1]):
                first, second = merged[0], merged[1]
                merged[1] = (first[0], second[0] + second[1] - first[0], second[2])
                merged.pop(0)
            if len(merged) >= 2 and is_edge_scoop(merged[-1], merged[-2]):
                last, prior = merged[-1], merged[-2]
                merged[-2] = (prior[0], last[0] + last[1] - prior[0], prior[2])
                merged.pop()
            return merged

        # every run is short (e.g. a brief staccato note): emit one note at the mode
        valid = window_notes[~np.isnan(window_notes)]
        if len(valid) == 0:
            return []
        mode, _ = stats.mode(valid)
        return [(0, len(window_notes), float(mode))]

    @staticmethod
    def _nan_median_smooth(values, kernel=5):
        # nan-aware rolling median to suppress single-frame pitch flickers
        half = kernel // 2
        out = values.copy()
        for j in range(len(values)):
            seg = values[max(0, j - half): j + half + 1]
            seg = seg[~np.isnan(seg)]
            if len(seg):
                out[j] = np.round(np.median(seg))
        return out

    @classmethod
    def audio_to_pitch(cls, sampling_rate, audio_vec, nperseg=256, nfft=4096):
        # nfft >> nperseg zero-pads each frame: at nfft=nperseg the frequency
        # bins (62.5 Hz at fs=16k) are wider than a semitone below ~1 kHz, so
        # detected pitches quantize to the wrong note
        audio_vec = cls._butter_bandpass_filter(audio_vec, MIN_WHISTLE_FREQ, MAX_WHISTLE_FREQ, sampling_rate)
        freq_vec, time_vec, spec_mtx = spectrogram(audio_vec, fs=sampling_rate, nperseg=nperseg, nfft=nfft)
        pitch = freq_vec[spec_mtx.argmax(axis=0)]
        dt = time_vec[1] - time_vec[0] if len(time_vec) > 1 else MIN_WHISTLE_DURATION
        win_length = max(5, int(round(AMP_SMOOTHING_DURATION / dt)) | 1)
        win_length = min(win_length, len(time_vec) if len(time_vec) % 2 == 1 else len(time_vec) - 1)
        amp_vec = savgol_filter(spec_mtx.max(axis=0), win_length, 3)
        amp_vec[amp_vec < 1] = 1
        return time_vec, pitch, amp_vec

    @staticmethod
    def _butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b_a = butter(order, [low, high], btype='band')
        y = lfilter(b_a[0], b_a[1], data)
        return y

    @staticmethod
    def midi_note_to_key(midi_notes):
        if len(midi_notes) == 0:
            return np.array([]), np.array([])

        # MIDI convention: note 69 = A4, so octave = note // 12 - 1
        key_octave = [num // len(NOTES) - 1 for num in midi_notes]
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

    def to_midi_blob64(self):
        mfile = self.construct_midi_file()
        with io.BytesIO() as output_file:
            mfile.writeFile(output_file)
            output_file.seek(0)
            blob = base64.b64encode(output_file.read()).decode('utf-8')
        return blob


if __name__ == '__main__':
    sample_audio_path = '../../test/test_data/whistle_sample_stairwaytoheaven2.wav'
    sample_midi_output_path = '../../test/test_output/whistle_sample_stairwaytoheaven2.mid'
    WhistleDetector.from_wav(sample_audio_path).to_midi_file(sample_midi_output_path)
    # WhistleDetector.from_webm()

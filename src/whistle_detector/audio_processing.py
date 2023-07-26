import matplotlib.pyplot as plt
import numpy as np
from midiutil import MIDIFile
from scipy import stats
from scipy.io import wavfile
from scipy.signal import butter, lfilter, savgol_filter, find_peaks, spectrogram


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def audio_to_freq_stats(sampling_rate, audio_vec, plot=False):
    # given an audio vector return the amplitude and frequency for each frame of audio with the highest energy

    MAX_WHISTLE_FREQ = 5000
    MIN_WHISTLE_FREQ = 500
    audio_vec_filter = butter_bandpass_filter(audio_vec, MIN_WHISTLE_FREQ, MAX_WHISTLE_FREQ, sampling_rate)
    freq_vec, time_vec, spec_mtx = spectrogram(audio_vec_filter, fs=sampling_rate, nfft=256)
    max_freq_vec = freq_vec[spec_mtx.argmax(axis=0)]
    amp_vec = savgol_filter(spec_mtx.max(axis=0), min(51, len(time_vec) if len(time_vec) % 2 == 1 else len(time_vec) - 1), 3)
    amp_vec[amp_vec < 1] = 1

    if plot:
        plt.specgram(audio_vec_filter, Fs=sampling_rate, NFFT=256)
        plt.figure()
        for a, i in enumerate(spec_mtx.T):
            plt.plot(freq_vec, np.log(i + a / 10))
            plt.axvline(max_freq_vec[a])
            pass
    return time_vec, max_freq_vec, amp_vec


def whistle_to_midi_notes(sampling_rate, audio_vec, plot=False):
    MIN_WHISTLE_DURATION = .05
    MAX_MIDI_VOLUMN = 127
    time_vec, max_energy_freq, max_energy_smooth = audio_to_freq_stats(sampling_rate, audio_vec, plot)

    peak_idx, peak_property = find_peaks(max_energy_smooth, width=MIN_WHISTLE_DURATION / time_vec[0], prominence=100)
    if len(peak_idx) == 0:
        return [], [], [], []

    duration = peak_property['widths'] * time_vec[0]
    time = time_vec[peak_idx]
    volume = np.round(MAX_MIDI_VOLUMN * (np.log(peak_property['prominences']) / np.log(max(peak_property['prominences'])))).astype(int)
    raw_midi_note_numbers = frequencies_to_midi_note_num(max_energy_freq)

    midi_notes = []
    for i, w in zip(peak_idx, peak_property['widths']):
        hw = int(w / 2)
        # for duration of the whistle find the mode note
        mode, mode_count = stats.mode(raw_midi_note_numbers[max((i - hw), 0): min((i + hw), len(time_vec))])
        midi_notes.append(int(mode[0]))

    if plot:
        plt.figure()
        plt.plot(time_vec, max_energy_freq)
        ax2 = plt.gca().twinx()
        # ax2.plot(time_vec, max_energy, c='r')
        ax2.plot(time_vec, max_energy_smooth, c='g')
        plt.scatter(time_vec[peak_idx], max_energy_smooth[peak_idx], c='r', marker='x')
        plt.scatter(time_vec[peak_idx], peak_property['prominences'], c='r', marker='x')
        plt.show()

    return time, duration, midi_notes, volume


def whistle_to_midi_notes_frame(sampling_rate, audio_vec, plot=False):
    MAX_MIDI_VOLUMN = 127
    _, max_energy_freq, max_energy_smooth = audio_to_freq_stats(sampling_rate, audio_vec, plot)
    raw_midi_note_numbers = frequencies_to_midi_note_num(max_energy_freq)
    mode, _ = stats.mode(raw_midi_note_numbers)
    midi_note = int(mode[0])
    note_indices = raw_midi_note_numbers == midi_note
    note_energy = np.log(np.mean(max_energy_smooth[note_indices]))
    volume =np.round(MAX_MIDI_VOLUMN *  note_energy/12).astype(int)
    duration = sum(note_indices) / sampling_rate

    return duration, midi_note, volume


def midi_note_to_key(midi_note_numbers):
    if len(midi_note_numbers) == 0:
        return np.array([]), np.array([])
    num_notes = 12
    NOTES = np.array(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
    freq_octave = [num // num_notes for num in midi_note_numbers]
    freq_note = NOTES[[int(num % num_notes) for num in midi_note_numbers]]
    return freq_octave, freq_note


def frequencies_to_midi_note_num(frequency):
    # algorithm taken from https://stackoverflow.com/a/64505498

    # A4 = 440 Hz = 58 pitch index for midi
    REFERENCE_FREQUENCY = 440
    REFERENCE_MIDI_PITCH = 58

    # note index = log_o(freq/440) where o = 2^(1/12) the diff of frequency between consecutive notes
    note_multiplier = 2 ** (1 / 12)
    distance_from_known_note = np.round(np.log(frequency / REFERENCE_FREQUENCY) / np.log(note_multiplier))
    midi_note_number = REFERENCE_MIDI_PITCH + distance_from_known_note

    # if midi note is out of range it's likely a bad frame, set value to nan
    midi_note_number[midi_note_number < 0] = float('nan')

    return midi_note_number


def notes_to_midi(time, duration, notes, volume, output_path):
    track = 0
    channel = 0
    tempo = 60  # In BPM
    mfile = MIDIFile(1)  # One track, defaults to format 1 (tempo track
    # automatically created)
    mfile.addTempo(track, 0, tempo)
    for t, d, n, v in zip(time, duration, notes, volume):
        print(t, d, n, v)
        mfile.addNote(track, channel, n, t, d, v)
    with open(output_path, "wb") as output_file:
        mfile.writeFile(output_file)


def whistle_to_midi(input_audio, output_midi):
    fs, audio = wavfile.read(input_audio)
    time, duration, midi_notes, volume = whistle_to_midi_notes(fs, audio, plot=False)
    notes_to_midi(time, duration, midi_notes, volume, output_midi)


if __name__ == '__main__':
    sample_audio_path = '../../test/test_data/whistle_sample_stairwaytoheaven2.wav'
    sample_midi_output_path = '../../test/test_output/whistle_sample_stairwaytoheaven2.mid'
    whistle_to_midi(sample_audio_path, sample_midi_output_path)

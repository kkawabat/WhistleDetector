import numpy as np
from midiutil import MIDIFile
from scipy import stats
from scipy.io import wavfile
from scipy.signal import spectrogram, butter, lfilter, savgol_filter, find_peaks
import matplotlib.pyplot as plt
import math
import mido


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


def whistle_to_notes(sampling_rate, audio_vec, plot=False):
    audio_vec_filter = butter_bandpass_filter(audio_vec, 500, 5000, sampling_rate)
    spec_mtx, freq_vec, time_vec, img = plt.specgram(audio_vec_filter, Fs=sampling_rate, NFFT=256)
    max_energy = spec_mtx.max(axis=0)
    max_energy_smooth = savgol_filter(max_energy, 51, 3)
    max_energy_freq = freq_vec[spec_mtx.argmax(axis=0)]

    peak_idx, peak_property = find_peaks(max_energy_smooth, width=.05/time_vec[0], prominence=100)
    duration = peak_property['widths'] * time_vec[0]
    time = time_vec[peak_idx]
    volume = np.round(127*(np.log(peak_property['prominences'])/np.log(max(peak_property['prominences'])))).astype(int)
    raw_midi_note_numbers = frequencies_to_midi_note_num(max_energy_freq)

    midi_notes = []
    for i, w in zip(peak_idx, peak_property['widths']):
        hw = int(w/2)
        # for duration of whistle find the mode note
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

    # return time_vec, max_energy_freq, note_name, note_octave
    return time, duration, midi_notes, volume


def frequencies_to_note(frequency):
    # algorithm taken from https://stackoverflow.com/a/64505498

    # define constants that control the algorithm
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']  # these are the 12 notes in each octave
    OCTAVE_MULTIPLIER = 2  # going up an octave multiplies by 2
    KNOWN_NOTE_NAME, KNOWN_NOTE_OCTAVE, KNOWN_NOTE_FREQUENCY = ('A', 4, 440)  # A4 = 440 Hz

    # calculate the distance to the known note since notes are spread evenly, going up a note will multiply by a
    # constant so we can use log to know how many times a frequency was multiplied to get from the known note to our
    # note this will give a positive integer value for notes higher than the known note, and a negative value for
    # notes lower than it (and zero for the same note)
    note_multiplier = OCTAVE_MULTIPLIER ** (1 / len(NOTES))
    frequency_relative_to_known_note = frequency / KNOWN_NOTE_FREQUENCY
    distance_from_known_note = np.log(frequency_relative_to_known_note) / np.log(note_multiplier)

    # round to make up for floating point inaccuracies
    distance_from_known_note = np.round(distance_from_known_note)

    # using the distance in notes and the octave and name of the known note, we can calculate the octave and name of
    # our note NOTE: the "absolute index" doesn't have any actual meaning, since it doesn't care what its zero point
    # is. it is just useful for calculation
    known_note_index_in_octave = NOTES.index(KNOWN_NOTE_NAME)
    known_note_absolute_index = KNOWN_NOTE_OCTAVE * len(NOTES) + known_note_index_in_octave
    note_absolute_index = known_note_absolute_index + distance_from_known_note
    note_octave, note_index_in_octave = note_absolute_index // len(NOTES), (note_absolute_index % len(NOTES)).astype(int)

    tmp_notes = NOTES + [""]  # some frames have invalid notes thus set as ""
    note_index_in_octave[note_index_in_octave < 0] = -1  # invalid index are set to -1 which loops to ""
    note_name = np.array(tmp_notes)[note_index_in_octave]
    return note_name, note_octave


def frequencies_to_midi_note_num(frequency):
    # algorithm taken from https://stackoverflow.com/a/64505498

    # define constants that control the algorithm
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']  # these are the 12 notes in each octave
    OCTAVE_MULTIPLIER = 2  # going up an octave multiplies by 2
    KNOWN_NOTE_NAME, KNOWN_NOTE_OCTAVE, KNOWN_NOTE_FREQUENCY = ('A', 4, 440)  # A4 = 440 Hz

    # calculate the distance to the known note since notes are spread evenly, going up a note will multiply by a
    # constant so we can use log to know how many times a frequency was multiplied to get from the known note to our
    # note this will give a positive integer value for notes higher than the known note, and a negative value for
    # notes lower than it (and zero for the same note)
    note_multiplier = OCTAVE_MULTIPLIER ** (1 / len(NOTES))
    frequency_relative_to_known_note = frequency / KNOWN_NOTE_FREQUENCY
    distance_from_known_note = np.log(frequency_relative_to_known_note) / np.log(note_multiplier)

    # round to make up for floating point inaccuracies
    distance_from_known_note = np.round(distance_from_known_note)

    # using the distance in notes and the octave and name of the known note, we can calculate the octave and name of
    # our note NOTE: the "absolute index" doesn't have any actual meaning, since it doesn't care what its zero point
    # is. it is just useful for calculation
    known_note_index_in_octave = NOTES.index(KNOWN_NOTE_NAME)
    known_note_absolute_index = KNOWN_NOTE_OCTAVE * len(NOTES) + known_note_index_in_octave
    note_absolute_index = known_note_absolute_index + distance_from_known_note
    note_octave, note_index_in_octave = note_absolute_index // len(NOTES), (note_absolute_index % len(NOTES)).astype(int)
    midi_note_number = (note_octave + 1) * 12 + note_index_in_octave
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
        mfile.addNote(track, channel, n, t, d, v)
    with open(output_path, "wb") as output_file:
        mfile.writeFile(output_file)


if __name__ == '__main__':
    fs, audio = wavfile.read('../test/test_data/whistle_sample_stairwaytoheaven.wav')
    time, duration, midi_notes, volume = whistle_to_notes(fs, audio, plot=False)
    sample_output_path = '../test/test_output/whistle_sample_16k.mid'
    notes_to_midi(time, duration, midi_notes, volume, sample_output_path)

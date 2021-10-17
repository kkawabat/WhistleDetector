from scipy.io import wavfile

from whilstle2midi.note_detector import notes_to_midi, whistle_to_notes


def run_from_audio_file():
    sample_input_audio_path = './test_data/whistle_sample_stairwaytoheaven.wav'
    sample_output_mid_path = './test_output/whistle_sample_16k.mid'
    fs, audio = wavfile.read(sample_input_audio_path)
    time, duration, midi_notes, volume = whistle_to_notes(fs, audio, plot=False)
    notes_to_midi(time, duration, midi_notes, volume, sample_output_mid_path)


if __name__ == '__main__':
    run_from_audio_file()

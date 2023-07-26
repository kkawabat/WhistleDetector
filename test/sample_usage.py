from src.whistle_detector.audio_processing import whistle_to_midi


def audio_file_to_midi_file():
    sample_input_audio_path = './test_data/whistle_sample_stairwaytoheaven.wav'
    sample_output_mid_path = './test_output/whistle_sample_stairwaytoheaven.mid'
    whistle_to_midi(sample_input_audio_path, sample_output_mid_path)

    sample_input_audio_path = './test_data/whistle_sample_walklikeegyptian.wav'
    sample_output_mid_path = './test_output/whistle_sample_walklikeegyptian.mid'
    whistle_to_midi(sample_input_audio_path, sample_output_mid_path)


if __name__ == '__main__':
    audio_file_to_midi_file()

from src.whistle_detector.audio_processing import WhistleDetector


def audio_file_to_midi_file():
    sample_input_audio_path = './test_data/whistle_sample_stairwaytoheaven.wav'
    sample_output_mid_path = './test_output/whistle_sample_stairwaytoheaven.mid'
    WhistleDetector.from_wav(sample_input_audio_path).to_midi_file(sample_output_mid_path)

    sample_input_audio_path = './test_data/whistle_sample_walklikeegyptian.wav'
    sample_output_mid_path = './test_output/whistle_sample_walklikeegyptian.mid'
    WhistleDetector.from_wav(sample_input_audio_path).to_midi_file(sample_output_mid_path)


def test_webm():
    sample_input_audio_path = r'.\test_data\sample.weba'
    WhistleDetector.from_webm(sample_input_audio_path)


def test_midi_blob():
    wd = WhistleDetector.from_wav(r'.\test_data\whistle_sample_stairwaytoheaven.wav')
    blob = wd.to_midi_blob64()
    print(blob)


if __name__ == '__main__':
    test_midi_blob()

from src.whistle_detector.audio_processing import WhistleDetector


def audio_file_to_midi_file():
    sample_input_audio_path = './test_data/whistle_sample_stairwaytoheaven.wav'
    sample_output_mid_path = './test_output/whistle_sample_stairwaytoheaven.mid'
    WhistleDetector.from_wav(sample_input_audio_path).to_midi_file(sample_output_mid_path)

    sample_input_audio_path = './test_data/whistle_sample_walklikeegyptian.wav'
    sample_output_mid_path = './test_output/whistle_sample_walklikeegyptian.mid'
    WhistleDetector.from_wav(sample_input_audio_path).to_midi_file(sample_output_mid_path)


def test_webm():
    sample_input_audio_path = r'C:\Users\kkawa\PycharmProjects\WhistleDetector\test\test_data\download.weba'
    WhistleDetector.from_webm(sample_input_audio_path)


def test_webm_blob():
    with open(r'C:\Users\kkawa\PycharmProjects\WhistleDetector\test\test_data\webm_blob', 'rb') as ifile:
        WhistleDetector.from_webm_blob(ifile)


def test_midi_blob():
    wd = WhistleDetector.from_wav(r'C:\Users\kkawa\PycharmProjects\WhistleDetector\test\test_data\whistle_sample_stairwaytoheaven.wav')
    blob = wd.to_midi_blob()
    pass


if __name__ == '__main__':
    test_midi_blob()


# ffmpeg -f lavfi -i testsrc=duration=10:size=1280x720:rate=30 testsrc.webm
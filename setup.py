from setuptools import setup, find_packages

setup(
    name='whistle_detector',
    version='1.0.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/kkawabat/WhistleDetector',
    license='MIT',
    author='Kan Kawabata',
    author_email='kkawabat@asu.edu',
    description='This project tries to convert your whistle into a midi file',
    install_requires=[
        'kaudio-library @ https://github.com/kkawabat/KaudioLibrary.git',
        'matplotlib',
        'MIDIUtil',
        'numpy',
        'PyAudio',
        'scipy',
        'sounddevice',
        'ffmpeg-python'
    ]
)

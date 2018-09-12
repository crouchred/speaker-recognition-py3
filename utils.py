from scipy.io import wavfile

def read_wav(fname):
    fs, signal = wavfile.read(fname)
    if len(signal.shape) != 1:
        print("convert stereo to mono")
        signal = signal[:,0]
    return fs, signal

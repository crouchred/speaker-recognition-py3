from python_speech_features import mfcc
import numpy as np

def get_feature(fs, signal):
    mfcc_feature = mfcc(signal, fs)
    #lpc = LPC.extract(tup)
    if len(mfcc_feature) == 0:
        print >> sys.stderr, "ERROR.. failed to extract mfcc feature:", len(signal)
    #return np.concatenate((mfcc, lpc), axis=1)
    return mfcc_feature

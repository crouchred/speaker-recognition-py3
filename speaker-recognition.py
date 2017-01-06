"""
../venv/bin/python speaker-recognition.py -t predict -i "~/tmp/new_album_voice_processed/13587302.wav" -m model.out
../venv/bin/python speaker-recognition.py -t enroll -i "~/tmp/speaker/*" -m model.test.out
"""
import os
import itertools
import glob
from utils import read_wav
from interface import ModelInterface

def task_enroll(input_dirs, output_model=None):
    m = ModelInterface()
    input_dirs = [os.path.expanduser(k) for k in input_dirs.strip().split()]
    dirs = itertools.chain(*(glob.glob(d) for d in input_dirs))
    dirs = [d for d in dirs if os.path.isdir(d)]

    files = []
    if len(dirs) == 0:
        print ("No valid directory found!")
        sys.exit(1)
    
    dirN = 0
    for d in dirs:
        if dirN > 3000:
            break
        dirN = dirN + 1
        label = os.path.basename(d.rstrip('/'))
        print(str(dirN) + ":" + label)

        wavs = glob.glob(d + '/*.wav')

        if len(wavs) == 0:
            print ("No wav file found in {0}".format(d))
            continue
        for wav in wavs:
            print(wav)
            fs, signal = read_wav(wav)
            print(fs)
            m.enroll(label, fs, signal)
            m.train()
            m.dump("test.out")
            #try:
            #except Exception as e:
            #    print(wav + " :error")
            #    print(e)

def task_predict(input_files, input_model):
    m = ModelInterface.load(input_model)
    for f in glob.glob(os.path.expanduser(input_files)):
        fs, signal = read_wav(f)
        label = m.predict(fs, signal)
        print (f, '->', label)

if __name__ == "__main__":
    task_enroll("~/tmp/speaker/*")
    task_predict("~/tmp/new_album_voice_processed/13587302.wav", "test.out")


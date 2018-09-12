### About
This project is a simple python3 version of [speaker-recognition](https://github.com/ppwwyyxx/speaker-recognition) and I make a little change for the convenience of command line usage.

### difference with speaker-recognition of python2
+ Neither use MFCC implementation of bob nor implement that myself. Use the [python_speech_features](https://github.com/jameslyons/python_speech_features) instead.
+ Remove the GUI and you can only use the command line to train and predict the model. 
+ Replace the function and class in sklearn which will be removed in the later version.
+ Use softmax function to output the probability.
+ convert to mono if the origin audio if stereo.

### Usage
```sh
usage: speaker-recognition.py [-h] -t TASK -i INPUT -m MODEL

Speaker Recognition Command Line Tool

optional arguments:
  -h, --help            show this help message and exit
  -t TASK, --task TASK  Task to do. Either "enroll" or "predict"
  -i INPUT, --input INPUT
                        Input Files(to predict) or Directories(to enroll)
  -m MODEL, --model MODEL
                        Model file to save(in enroll) or use(in predict)

Wav files in each input directory will be labeled as the basename of the directory.
Note that wildcard inputs should be *quoted*, and they will be sent to glob module.

Examples:
    Train:
    ./speaker-recognition.py -t enroll -i "/tmp/person* ./mary" -m model.out

    Predict:
    ./speaker-recognition.py -t predict -i "./*.wav" -m model.out
```

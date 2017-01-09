### About
This project is a simple python3 version of [speaker-recognition](https://github.com/ppwwyyxx/speaker-recognition) 

### difference with speaker-recognition of python2
For the convenience, I removed a lot of features I didn't use.
+ Neither use MFCC implementation of bob nor implement that myself. Use the [python_speech_features](https://github.com/jameslyons/python_speech_features) instead.
+ Remove the GUI and you can only use the command line to train and predict the model. 
+ Replace the function and class in sklearn which will be removed in the later version.

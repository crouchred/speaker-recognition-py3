import pickle
from collections import defaultdict
from skgmm import GMMSet
from features import get_feature
import time

class ModelInterface:

    def __init__(self):
        self.features = defaultdict(list)
        self.gmmset = GMMSet()

    def enroll(self, name, fs, signal):
        feat = get_feature(fs, signal)
        self.features[name].extend(feat)

    def train(self):
        #self.gmmset = self._get_gmm_set()
        self.gmmset = GMMSet()
        start_time = time.time()
        for name, feats in self.features.items():
            self.gmmset.fit_new(feats, name)
            #try:
            #except Exception as e :
            #    print ("%s failed"%(name))
        print (time.time() - start_time, " seconds")

    def dump(self, fname):
        """ dump all models to file"""
        self.gmmset.before_pickle()
        with open(fname, 'wb') as f:
            pickle.dump(self, f, -1)
        self.gmmset.after_pickle()

    def predict(self, fs, signal):
        """
        return a label (name)
        """
        feat = get_feature(fs, signal)
        #try:
        #except Exception as e:
        #    print (e)
        import ipdb;ipdb.set_trace()
        return self.gmmset.predict_one(feat)

    @staticmethod
    def load(fname):
        """ load from a dumped model file"""
        with open(fname, 'rb') as f:
            R = pickle.load(f)
            R.gmmset.after_pickle()
            return R

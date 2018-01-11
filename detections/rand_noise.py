from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.activations import softmax


class RandNoiseDetector:
    def __init__(self, model, detector_name):
        self.model = model
        self.noises = []

        #print("model.layers[0]", model.layers[0].batch_input_shape)
        shape = model.layers[0].batch_input_shape[1:]
        #print("shape:", shape)

        np.random.seed(1234)
        num = 100
        for _ in range(num):
            n = np.random.randn(*shape) * 0.2
            #n = np.random.randn(*shape) * 0.1
            self.noises.append( n )

    def train(self, X=None, Y=None, fpr=None):
        #print("RandNoiseDetector does not require training, do nothing.")
        pass

    def test(self, X):
        #print ("X.shape: ", X.shape)

        is_adv= []
        xs = list(X)
        for x in xs:
            xs_with_noise = np.array( [ x + n for n in self.noises] )
            scores = self.model.predict(xs_with_noise)
            labels = np.argmax(scores, axis=1)
            #is_adv.append(  np.all( labels == labels[0] ) )
            ct = np.bincount( labels )
            majority = np.argmax( ct )
            is_adv.append( ct[majority] / len(labels) )

        #print("is_adv:", is_adv)
        res = np.array([ 0 if x > 0.99 else 1 for x in is_adv ] )
        #print("res:", res)
        return res,is_adv


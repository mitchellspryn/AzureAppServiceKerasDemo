"""
Definition of views.
"""

from django.shortcuts import render
from django.http import HttpRequest, JsonResponse
from django.template import RequestContext
from datetime import datetime
from django.views.decorators.csrf import *
from django.http import HttpResponse

from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Input, Dropout
from keras.initializers import RandomUniform
from keras.optimizers import SGD
from keras.models import model_from_json
import keras.backend as K

import tensorflow as tf

import numpy as np
import pandas as pd
import h5py
import json
import timeit

import os
print(os.getcwd())

K.set_learning_phase(0)

#Load the model from disk
with open('model.json', 'r') as f:
    loaded_model = model_from_json(f.read())

loaded_model.load_weights('model.h5')
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print('loaded model')

#Save the default graph for use in another thread.
# This is a tensorflow quirk.
graph = tf.get_default_graph()

@csrf_exempt
def predict(request):
    #Re-load the model onto the current request thread.
    global graph
    with graph.as_default():
        
        #Check for POST usage
        if request.method != 'POST':
            return 'Need post please.'

        #Decode the data
        data = json.loads(request.body.decode('utf-8'))
        X = np.array(list(map(lambda r: r.split(' '), data['X'].split('|'))))

        #Predict
        labels = loaded_model.predict(X, verbose=0)

        #Return the response
        return JsonResponse({'labels': labels.tolist()})
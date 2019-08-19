#!/usr/bin/env python
from sklearn.externals import joblib
import numpy as np
import pandas as pd

def get_sepsis_score(data, model):

    M1 = joblib.load('model-saved.pkl')
    s_m = np.load('septic_mean.npy', allow_pickle=True)
    ns_m = np.load('Nonseptic_mean.npy', allow_pickle=True)
    All = np.vstack((s_m, ns_m))
    maenAll = np.mean(All, axis=0)

    for i in range(data.shape[1]):
        if np.isnan(data[0, i]):
            data[0, i] = maenAll[i]


    df = pd.DataFrame.from_records(data)

    df.interpolate(method='linear', inplace=True)
    data = np.array(df)
    for column in range(data.shape[1]):
        col = data[:, column]
        value = col[np.isnan(col)]
        if len(value) > 0:
            col[np.isnan(col)] = maenAll[column]
        data[:, column] = col

    predicted = M1.predict(data)

    score = np.random.rand(len(data), 1)
    for i in range(len(data)):
        if predicted[i]==0:
         score[i] = 0.4
        else:
         score[i] = 0.6

    label = np.copy(predicted)

    return score, label

def load_sepsis_model():

    return None

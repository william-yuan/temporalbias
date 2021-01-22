import pyodbc
import pandas as pd
import time
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from tqdm import tqdm, tqdm_notebook
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

store = pd.HDFStore('2015_44andunder_completecases.h5')
seq = store['seq'] 
membertable = store['membertable']
deliveryPhe = ['661', '665', '650', '636.2', '663', '669', '668', '656.1', '652', '658']
deliveryCPT = ['92585','99462', '01961','99460','01967', '01968','59409', '59410', '59414', '59514','59612','59620', '59515', '59614', '59622', '59025'] 
phemems = seq[seq['Object'].isin(deliveryPhe)]['MemberNum'].unique()
cptmems = seq[seq['Object'].isin(deliveryCPT)]['MemberNum'].unique()

cases = np.unique(np.concatenate([cptmems, phemems]))
cases = pd.DataFrame(cases, columns=['MemberNum'])
seq['Date'] = pd.to_datetime(seq['Date'])
seq['Day'] = (seq['Date'] - pd.to_datetime('2015-01-01')).dt.days
cases['StartDate'] = 'abc'
cases['Day'] = -1

for i in tqdm_notebook(range(30000)):
    df = seq[seq['MemberNum'] == cases['MemberNum'][i]]
    df = df[df['Object'].isin(deliveryCPT + deliveryPhe)]
    if df.shape[0] > 1:
        cases['StartDate'][i] = df['Date'].min()
        cases['Day'][i] = df['Day'].min()

seq=seq.merge(membertable[['MemberNum', 'BirthYearMinus1900']], on = 'MemberNum')
cases=cases.merge(membertable[['MemberNum', 'BirthYearMinus1900']], on = 'MemberNum')
cases = cases[cases['Day'] >= 30].reset_index(drop = True)
controlpool = membertable[~membertable['MemberNum'].isin(cases['MemberNum'])]
matched_controls = cases.copy()
for i in tqdm_notebook(range(cases.shape[0])):
    df = controlpool[controlpool['BirthYearMinus1900'] == matched_controls['BirthYearMinus1900'][i]]['MemberNum']
    matched_controls['MemberNum'][i] = df.sample(n=1).values[0]

seqCases = seq[seq['MemberNum'].isin(cases['MemberNum'])]
seqControl = seq[seq['MemberNum'].isin(matched_controls['MemberNum'])]

for i in tqdm_notebook(range(cases.shape[0])):
    member = cases['MemberNum'][i]
    indexdate = cases['Day'][i]
    seqCases = seqCases.drop(seqCases[(seqCases['MemberNum'] == member) & ~(seqCases['Day'].between(0,30, inclusive = True))] .index).reset_index(drop = True)
    
for i in tqdm_notebook(range(matched_controls.shape[0])):
    member = matched_controls['MemberNum'][i]
    indexdate = matched_controls['Day'][i]
    seqControl = seqControl.drop(seqControl[(seqControl['MemberNum'] == member) & ~(seqControl['Day'].between(0,30, inclusive = True))] .index)
    
seqCases['Type'] = seqCases['Type'].astype('int64')
seqCases['Object'] = seqCases['Object'].astype(str)
seqCases['ObjVal'] = pd.to_numeric(seqCases['ObjVal'], errors='coerce')
seqCases['Date'] = pd.to_datetime(seqCases['Date'])

seqControl['Type'] = seqControl['Type'].astype('int64')
seqControl['Object'] = seqControl['Object'].astype(str)
seqControl['ObjVal'] = pd.to_numeric(seqControl['ObjVal'], errors='coerce')
seqControl['Date'] = pd.to_datetime(seqControl['Date'])

seqControl['Type_obj'] = seqControl['Type'].astype(str) + '_' + seqControl['Object'].astype(str)
seqCases['Type_obj'] = seqCases['Type'].astype(str) + '_' + seqCases['Object'].astype(str)

seqCases['Type_obj'] = seqCases['Type_obj'].str.strip()
seqControl['Type_obj'] = seqControl['Type_obj'].str.strip()

fname = "./model_genpop_embed.model"

vecs = Word2Vec.load(fname)
seqControl_filtered = seqControl[seqControl['Type_obj'].isin(vecs.wv.vocab)]
seqCases_filtered = seqCases[seqCases['Type_obj'].isin(vecs.wv.vocab)]

wordsCases = seqCases_filtered['Type_obj'].as_matrix()
vec_listCases = []
for w in tqdm_notebook(wordsCases):
    vec_listCases.append(vecs.wv.get_vector(w))
seqCases_filtered['vector'] = vec_listCases

wordsControls = seqControl_filtered['Type_obj'].as_matrix()
vec_listControls = []
for w in tqdm_notebook(wordsControls):
    vec_listControls.append(vecs.wv.get_vector(w))
seqControl_filtered['vector'] = vec_listControls

from sklearn.utils import shuffle
size = 20
controlSeqs = []
controlID = []
controlTarg = []
controlGuide = []

for m in tqdm_notebook(seqControl_filtered.MemberNum.unique()):
    seqSubject = seqControl_filtered[seqControl_filtered['MemberNum'] == m]
    img_arr = np.stack(seqSubject['vector'].values)
    guide = seqSubject['Type_obj']
    if img_arr.shape[0] > size:
        img_arr = img_arr[-size:,:]
        guide = guide[-size:]
    elif img_arr.shape[0] < (size*1):
        img_arr = np.pad(img_arr, ((0,size - img_arr.shape[0]),(0,0)), 'constant')

    img_arr = np.pad(img_arr, ((5,0), (0,0)), 'constant')   
    controlSeqs.append(img_arr.astype('float16'))
    controlTarg.append(0)    
    controlID.append(m)
    controlGuide.append(guide)

casesSeqs = []
casesTarg = []
casesID = []
casesGuide = []

for m in tqdm_notebook(seqCases_filtered.MemberNum.unique()[:len(controlSeqs)]):
    seqSubject = seqCases_filtered[seqCases_filtered['MemberNum'] == m]
    img_arr = np.stack(seqSubject['vector'].values)
    guide = seqSubject['Type_obj']
    if img_arr.shape[0] > size:
        img_arr = img_arr[-size:,:]
        guide = guide[-size:]
    elif img_arr.shape[0] < (size*1):
        img_arr = np.pad(img_arr, ((0,size - img_arr.shape[0]),(0,0)), 'constant')

    img_arr = np.pad(img_arr, ((5,0), (0,0)), 'constant')   
    casesSeqs.append(img_arr.astype('float16'))
    casesTarg.append(1)    
    casesID.append(m)
    casesGuide.append(guide)
set(map(np.shape, controlSeqs))
set(map(np.shape, casesSeqs))

testratio = 0.85
ccmultiplier=1

import random
controlComb = list(zip(controlSeqs, controlID, controlGuide))
random.shuffle(controlComb)
casesComb = list(zip(casesSeqs, casesID, casesGuide))
random.shuffle(casesComb)
control = list(zip(controlComb, controlTarg))
cases = list(zip(casesComb, casesTarg))

trainsize = int(round(len(casesSeqs)* testratio))
train = control[:(int(ccmultiplier)*trainsize)] + cases[:trainsize]
test = control[(int(ccmultiplier)*trainsize):] + cases[trainsize:]
random.shuffle(train)



random.shuffle(test)
trainComb, trainTarg = zip(*train)
testComb, testTarg = zip(*test)

trainSeqs, trainID, trainGuide = zip(*trainComb)
testSeqs, testID, testGuide = zip(*testComb)
testSeqs = np.array(testSeqs)
testTarg = np.array(testTarg)
trainTarg = np.array(trainTarg)
trainSeqs = np.array(trainSeqs)

seq_length = 25


model = Sequential()

model.add(Conv1D(4, 1, activation='relu', input_shape=(seq_length, 100)))
model.add(MaxPooling1D(3))
model.add(Dropout(0.5))
model.add(Conv1D(4, 1, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense (4, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              
              metrics=['accuracy'])

history = model.fit(trainSeqs, trainTarg, batch_size=50, epochs=100, validation_data=(testSeqs, testTarg), class_weight= {0: 1, 1: ccmultiplier})


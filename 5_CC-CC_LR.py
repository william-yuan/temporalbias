import pyodbc
import pandas as pd
import time
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from tqdm import tqdm, tqdm_notebook

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

cases = cases_master[cases_master['Day'] >= 180].reset_index(drop = True)
controlpool = membertable[~membertable['MemberNum'].isin(cases['MemberNum'])]

for i in tqdm_notebook(range(cases.shape[0])):
    df = controlpool[controlpool['BirthYearMinus1900'] == matched_controls['BirthYearMinus1900'][i]]['MemberNum']
    matched_controls['MemberNum'][i] = df.sample(n=1).values[0]
seqCases = seq[seq['MemberNum'].isin(cases['MemberNum'])]
seqControl = seq[seq['MemberNum'].isin(matched_controls['MemberNum'])]

for i in tqdm_notebook(range(cases.shape[0])):
    member = cases['MemberNum'][i]
    indexdate = cases['Day'][i]
    seqCases = seqCases.drop(seqCases[(seqCases['MemberNum'] == member) & ~(seqCases['Day'].between(indexdate-120, indexdate-90, inclusive = True))] .index).reset_index(drop = True)
    
for i in tqdm_notebook(range(matched_controls.shape[0])):
    member = matched_controls['MemberNum'][i]
    indexdate = matched_controls['Day'][i]
    seqControl = seqControl.drop(seqControl[(seqControl['MemberNum'] == member) & ~(seqControl['Day'].between(indexdate-120, indexdate-90, inclusive = True))] .index)
    
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

seqCases['d'] = 1
Cases_LR=seqCases.pivot_table(index='MemberNum', values= 'd', columns='Type_obj',aggfunc=np.sum)
Cases_LR[Cases_LR.isnull()] = 0

seqControl['d'] = 1
Control_LR=seqControl.pivot_table(index='MemberNum', values= 'd', columns='Type_obj',aggfunc=np.sum)
Control_LR[Control_LR.isnull()] = 0

for i in Control_LR.columns.difference(Cases_LR.columns):
    Cases_LR[i] = 0

for i in Cases_LR.columns.difference(Control_LR.columns):
    Control_LR[i] = 0
Cases_LR = Cases_LR[Control_LR.columns]

Cases_LR = Cases_LR.sample(frac=1).reset_index(drop=True)
Control_LR = Control_LR.sample(frac=1).reset_index(drop=True)
Cases_LR = Cases_LR[:Control_LR.shape[0]]
Cases_LR_backup = Cases_LR.sample(frac=1)
Control_LR_backup = Control_LR.sample(frac=1)

testratio = 0.85
ccmultiplier = 1
Cases_LR_backup['y'] = 1
Control_LR_backup['y'] = 0
trainsize = int(round(Cases_LR_backup.shape[0]*testratio))
train_data = pd.concat([Control_LR_backup[:(int(ccmultiplier)*trainsize)], Cases_LR_backup[:trainsize]])
test_data = pd.concat([Control_LR_backup[(int(ccmultiplier)*trainsize):Control_LR_backup.shape[0]], Cases_LR_backup[trainsize:Cases_LR_backup.shape[0]]])
test_data = test_data.sample(frac=1)
train_data = train_data.sample(frac=1)
test_targets = list(test_data['y'])
train_targets = list(train_data['y'])
train_data = train_data.drop(['y'], axis=1)
test_data = test_data.drop(['y'], axis=1)

#logistic regression 
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(solver = 'lbfgs', verbose = True, max_iter=10000)
lg.fit(train_data, train_targets)
test_pred = lg.predict(test_data)
roc_auc_score(test_targets, test_pred)


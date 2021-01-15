import pyodbc
import pandas as pd
import time
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from tqdm import tqdm, tqdm_notebook
cn = pyodbc.connect('')
user = ''
cursor = cn.cursor()

store = pd.HDFStore('2015_44andunder.h5')
seq = store['seq'] 
membertable = store['membertable']
membertable = membertable.fillna(0).astype(int)
deliveryPhe = ['650', '636.2', '663', '669', '668']
deliveryCPT = ['59409', '59410', '59414', '59514','59612','59620', '59515', '59614', '59622'] 
globalCPT = ['59400', '59510', '59610', '59618']

phemems = seq[seq['Object'].isin(deliveryPhe)]['MemberNum'].unique()
cptmems = seq[seq['Object'].isin(deliveryCPT)]['MemberNum'].unique()
globalmems = seq[seq['Object'].isin(globalCPT)]['MemberNum'].unique()

children = pd.read_sql("select * from MEMBERTABLE where BirthYearMinus1900 >= 115 and BirthYearMinus1900 < 116", cn)

birthmothers_s = membertable[membertable['SubscriberNum'].isin(children['SubscriberNum'])]['MemberNum']
birthmothers_m = membertable[membertable['MemberNum'].isin(children['SubscriberNum'])]['MemberNum']
delivery = np.unique(np.concatenate([cptmems, phemems, globalmems, birthmothers_s, birthmothers_m]))
print(len(delivery),  seq['MemberNum'].unique().shape[0])


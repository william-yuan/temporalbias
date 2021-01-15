import pyodbc
import pandas as pd
import time
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from tqdm import tqdm, tqdm_notebook
cn = pyodbc.connect()
cursor = cn.cursor()
user = ""
table_prefix = 'TemporalBias'


### 
cursor.execute("IF OBJECT_ID('" + user + ".dbo." + table_prefix + "_members', 'U') " +  
               "IS NOT NULL DROP TABLE " + user + ".dbo." + table_prefix + "_members;")
member_query = ("SELECT top " + limit + " MemberNum, SubscriberNum, BirthYearMinus1900 INTO "
              + user + ".dbo." + table_prefix + "_members " + 
              "FROM gmw3.dbo.Member " +
              "WHERE BirthYearMinus1900 >= 71 AND Gender = 'F' AND MemberNum in (SELECT MemberNum from gmw3.dbo.MemberEnrollment WHERE NumberOfMonths2015 = 12)" +
              "GROUP BY MemberNum, SubscriberNum, BirthYearMinus1900 ORDER BY newid();")
cursor.execute(member_query)
membertable = pd.read_sql('select * from' + user + '.dbo.timedemo_members', cn)
###

###
inclusion_tables = "(SELECT MemberNum from " + user + ".dbo." + table_prefix + "_members)"

cursor.execute("IF OBJECT_ID('" + user + ".dbo." + table_prefix + "_diagnosis', 'U') " +  
               "IS NOT NULL DROP TABLE " + user + ".dbo." + table_prefix + "_diagnosis;")

cursor.execute("IF OBJECT_ID('" + user + ".dbo." + table_prefix + "_procedures', 'U') " +  
               "IS NOT NULL DROP TABLE " + user + ".dbo." + table_prefix + "_procedures;")
cursor.execute("IF OBJECT_ID('" + user + ".dbo." + table_prefix + "_phenotypes', 'U') " +  
               "IS NOT NULL DROP TABLE " + user + ".dbo." + table_prefix + "_phenotypes;")


diag_query = ("SELECT d.MemberNum MemberNum, d.StartDate Date, 0 Type, " + 
              "CAST(d.DiagnosisCode AS varchar(32)) AS  Object, CAST(d.DiagnosisRank AS varchar(32)) AS ObjVal, " + 
              "CAST(d.PresentOnAdmission AS varchar(32)) AS ObjVal2 " + 
              "INTO " + user + ".dbo." + table_prefix + "_diagnosis " +
              "FROM (select * from gmw3.dbo.ObservationDiagnosis where year(StartDate) = 2015) D WHERE d.MemberNum IN " +
              inclusion_tables)              
print(diag_query)
cursor.execute(diag_query)
diags = cursor.execute("SELECT COUNT(1) FROM " + user + ".dbo." +
                        table_prefix + "_diagnosis").fetchone()
print('diags: ', diags)

phe_query = ("SELECT d.MemberNum, d.Date, d.Type, " + 
              "CAST(p.PheWasCode AS varchar(32)) AS  Object, d.ObjVal, " + 
              "d.ObjVal2 " + 
              "INTO " + user + ".dbo." + table_prefix + "_phenotypes " +
              "FROM " + user + ".dbo." + table_prefix + "_diagnosis d INNER JOIN Phewas.dbo.Icd9CodeTranslation p on p.Icd9Code=d.Object" )              
print(phe_query)
cursor.execute(phe_query)
phenotypes = cursor.execute("SELECT COUNT(1) FROM " + user + ".dbo." +
                        table_prefix + "_phenotypes").fetchone()
print('phenotypes: ', phenotypes)
proc_query = ("SELECT p.MemberNum MemberNum, p.DateServiceStarted Date, 2 Type, " + 
              "CAST(p.LineLevelProcedureCode as varchar(32)) as Object, " +
              "CAST(p.LineLevelProcedureCodeModifier as varchar(32)) as ObjVal, " +
              "CAST(p.LineLevelProcedureCodeType as varchar(32)) as ObjVal2 " +
              "INTO " + user + ".dbo." + table_prefix + "_procedures " +
              "FROM (select * from gmw3.dbo.ObservationProcedure where year(DateServiceStarted) = 2015) p WHERE p.MemberNum IN " +
              inclusion_tables)              

print(proc_query)
cursor.execute(proc_query)
procs = cursor.execute("SELECT COUNT(1) FROM " + user + ".dbo." +
                        table_prefix + "_procedures").fetchone()
print('procs: ', procs)
###


###
cursor.execute("IF OBJECT_ID('" + user + ".dbo." + table_prefix + "_sequences', 'U') " +  
               "IS NOT NULL DROP TABLE " + user + ".dbo." + table_prefix + "_sequences;")

unify_query = ("SELECT MemberNum, Date, Type, " + 
               "Object=CONVERT(varchar(32), Object), " + 
               "ObjVal=CONVERT(varchar(32), ObjVal), " +
               "ObjVal2=CONVERT(varchar(32), ObjVal2) " +
               "INTO " + user + ".dbo." + table_prefix + "_sequences " + 
               "FROM (" + 
               "SELECT * FROM " + user + ".dbo." + table_prefix + "_phenotypes UNION " +
               "SELECT * FROM " + user + ".dbo." + table_prefix + "_procedures) unify;")

print(unify_query)
cursor.execute(unify_query)
sql = "SELECT * FROM " + user + ".dbo." + table_prefix + "_sequences ORDER BY MemberNum, Date, TYPE"
seq = pd.read_sql(sql, cn)
###

store = pd.HDFStore('2015_44andunder_completecases.h5')
store['seq'] = seq
store['membertable'] = membertable
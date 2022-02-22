# -*- coding: utf-8 -*-
"""
@author: Chengkui Zhao
"""
import lightgbm as lgb
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
from scipy.stats.stats import pearsonr,spearmanr
df=pd.read_csv(r'AllFeature.csv')
df=df.iloc[:,1:]
y=pd.read_csv(r'label.csv')
y=pd.Series(y.iloc[:,3])
###Data preprocessing
featurename=df.columns.tolist()
featurename_numeric=featurename[63:147]+featurename[330:504]
standardscaler=StandardScaler()
df[featurename_numeric]=standardscaler.fit_transform(df[featurename_numeric])
Index=list(range(18384))
kf=KFold(n_splits=5,random_state=5,shuffle=True)
def Pearsoncorrelation(preds, train_data):
    labels = train_data
    cor=pearsonr(labels, preds)[0] 
    return 'pearson', cor ,True
Prediction=[]
Y_test=[]
for Train,Test in kf.split(Index):
    categorical_columns=featurename[:63]+featurename[147:330]
    training_data = lgb.Dataset(df.iloc[Train,], label=y[Train])
    test_data=lgb.Dataset(df.iloc[Test,], label=y[Test])
###Model parameter
param = {'num_leaves': '20',
         'objective': 'regression',
         'metric':('rmse')}
evals_result = {}  #Record the training process
###Model training
num_round = 80
boostermodel = lgb.train(param, training_data, num_round,valid_sets=[training_data],
                         evals_result=evals_result,categorical_feature=categorical_columns)
###Model prediction on test data
predictions=boostermodel.predict(df.iloc[Test,])
Pearsoncorrelation(predictions,y[Test])























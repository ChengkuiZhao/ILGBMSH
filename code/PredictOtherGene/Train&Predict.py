# -*- coding: utf-8 -*-
"""
@author: Chengkui Zhao
"""
import lightgbm as lgb
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
import numpy as np
from scipy import interp
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv(r'AllFeature.csv')
df=df.iloc[:,1:]
label=pd.read_csv(r'label.csv')
y=pd.Series(label.iloc[:,2])
data=pd.read_csv(r'Feature_PDCD1.csv')
data=data.iloc[:,1:]
shRNAname=label.iloc[:,0]
###Data preprocessing
featurename=df.columns.tolist()
featurename_numeric=featurename[63:147]+featurename[330:504]
standardscaler=StandardScaler()
df[featurename_numeric]=standardscaler.fit_transform(df[featurename_numeric])
###Training model
X_train=df
y_train=y
categorical_columns=featurename[:63]+featurename[147:330]
training_data = lgb.Dataset(X_train, label=y_train)
param = {    'num_leaves': 200,
             'learning_rate':0.008,
             'objective' : 'binary',
             'min_data_in_leaf':100,
             'max_depth':4,
             'feature_fraction':0.02,
             'objective': 'binary',
             'metric':('average_precision'),
             'num_iteration':1000,
              'verbose':-1,
} 
boostermodel = lgb.train(param, training_data,valid_sets=training_data,
                         verbose_eval = 100,categorical_feature=categorical_columns)
###Prediction for other gene
prediction=boostermodel.predict(data)
pd.DataFrame(data=prediction).to_csv(r'prediction.csv',encoding='gbk')
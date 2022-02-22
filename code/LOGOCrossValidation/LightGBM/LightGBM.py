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
df=pd.read_csv(r'.\AllFeature.csv')
df=df.iloc[:,1:]
label=pd.read_csv(r'.\label.csv')
y=pd.Series(label.iloc[:,1])
patientname=label.iloc[:,0]
###Data preprocessing
featurename=df.columns.tolist()
featurename_numeric=featurename[63:147]+featurename[330:504]
standardscaler=StandardScaler()
df[featurename_numeric]=standardscaler.fit_transform(df[featurename_numeric])
cv=[]
cv.append(list(range(0,1892)))
cv.append(list(range(1892,4045)))
cv.append(list(range(4045,4542)))
cv.append(list(range(4542,8996)))
cv.append(list(range(8996,12387)))
cv.append(list(range(12387,14696)))
cv.append(list(range(14696,15905)))
cv.append(list(range(15905,16462)))
cv.append(list(range(16462,18125)))
###LOGO Cross Validation
Prediction=[]
aucs = []
mean_recall = np.linspace(0, 1, 100)
plt.figure(dpi=300,figsize=(15,12))
Eval_result=[{},{},{},{},{},{},{},{},{}]
for i in range(0, 9):
    X_train=df.drop(cv[i])
    y_train=y.drop(cv[i])
    X_val=df.iloc[cv[i]]
    y_val=y[cv[i]]
    categorical_columns=featurename[:63]+featurename[147:330]
    training_data = lgb.Dataset(X_train, label=y_train)
    validation_data = lgb.Dataset(X_val, label=y_val)
    param = {'num_leaves': 200,
              'learning_rate':0.008,
                 'objective' : 'binary',
                 'min_data_in_leaf':100,
                 'max_depth':4,
                 'num_trees':8000,
                 'feature_fraction':0.02,
                 'objective': 'binary',
                 'metric':('average_precision'),
                  'verbose':-1} 
    boostermodel = lgb.train(param, training_data,valid_sets=[training_data, validation_data],evals_result=Eval_result[i],
                             verbose_eval = 1000,categorical_feature=categorical_columns)
    prediction=boostermodel.predict(X_val)
    Prediction.extend(prediction)
    precision,recall,thresholds=metrics.precision_recall_curve(y_val,prediction)    
    prc_auc=metrics.auc(recall,precision)  
    aucs.append(prc_auc)   
    plt.plot(recall,precision,lw=1,label='Gene %d (auPR=%0.3f)' % (i+1, prc_auc))
precision,recall,thresholds=metrics.precision_recall_curve(y[:18125],Prediction)
prc_auc=metrics.auc(recall,precision)
plt.plot(recall,precision,lw=3,color='red',label='All Genes (auPR=%0.3f)' % ( prc_auc))
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall',fontsize=20)
plt.ylabel('Precision',fontsize=20)
plt.title('Precision Recall Curve',fontsize=30)
plt.legend(loc="lower left",fontsize=20)    
plt.savefig('PRC for All Genes.pdf')
###Plot ROC AUC
a=[]
b=Prediction
plt.figure(dpi=600,figsize=(15,12))
for i in range(0,9) :
    y_val=y[cv[i]]
    a.extend(y_val)
###Compute ROC curve and area the curve
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
plt.figure(dpi=300,figsize=(15,12))
for i in range(0,9) :
    y_val=y[cv[i]]
    fpr,tpr,thresholds=metrics.roc_curve(y_val,Prediction[cv[i][0]:(cv[i][len(cv[i])-1]+1)],pos_label=1)
    mean_tpr+=interp(mean_fpr,fpr,tpr)
    mean_tpr[0]=0.0
    roc_auc=metrics.auc(fpr,tpr)
    plt.plot(fpr,tpr,lw=1,label='Gene %d (auROC=%0.3f)' % (i+1, roc_auc))
fpr,tpr,thresholds=metrics.roc_curve(a,b,pos_label=1)
roc_auc=metrics.auc(fpr,tpr)
plt.plot(fpr,tpr,lw=3,color='red',label='All Genes (auROC=%0.3f)' % (roc_auc))
plt.plot([0,1],[0,1],'--',color=(0.6,0.6,0.6),label='luck')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate',fontsize=20)
plt.ylabel('True Positive Rate',fontsize=20)
plt.title('Receiver Operating characteristic ',fontsize=30)
plt.legend(loc="lower right",fontsize=20)
#plt.show()
plt.savefig('ROC for All Genes.pdf')
    














































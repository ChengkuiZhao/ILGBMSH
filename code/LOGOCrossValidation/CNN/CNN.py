# -*- coding: utf-8 -*-
"""
@author: Chengkui Zhao
"""
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tensorboardX import SummaryWriter
from torchsummary import summary
import math
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report
from sklearn import metrics
import seaborn as sns
import numpy as np
from scipy import interp
import graphviz
# Hyperparameters
num_epochs = 20
num_epochs=20
num_classes = 2
batch_size = 300
learning_rate = 0.0005
kernelsize1=4
kernelsize2=4
kernelsize3=4
sequence=pd.read_csv(r'.\sequence.csv')
sequence=sequence.iloc[:,1:].values.tolist()
label=pd.read_csv(r'.\label.csv')
y=np.array(label.iloc[:,1],dtype=np.double)
shRNAname=label.iloc[:,0]
###Data prerocessing
#One-hot
enc = OneHotEncoder(handle_unknown='ignore')
X=[['A'],['C'],['G'],['T']]
enc.fit(X)
Sequence=[]
for i in range(28042):
    b=[list(x) for x in sequence[i][0] ]
    c=enc.transform(b).toarray()
    Sequence.append(c)
Sequence=np.array(Sequence)
input=torch.from_numpy(Sequence)
###Extract 9 genes in TILE set
TILEname=['Bcl2','hMyc','Hras','Kras','Mcl1','Myc','PCNA','Rpa3','Trp53']
a=[str.split(name,sep='_')[0] for name in shRNAname]
###Extract index of 9 genes shRNA
cv=[]
cv.append([i for i in range(len(a)) if a[i] == TILEname[0]])
cv.append([i for i in range(len(a)) if a[i] == TILEname[1]])
cv.append([i for i in range(len(a)) if a[i] == TILEname[2]])
cv.append([i for i in range(len(a)) if a[i] == TILEname[3]])
cv.append([i for i in range(len(a)) if a[i] == TILEname[4]])
cv.append([i for i in range(len(a)) if a[i] == TILEname[5]])
cv.append([i for i in range(len(a)) if a[i] == TILEname[6]])
cv.append([i for i in range(len(a)) if a[i] == TILEname[7]])
cv.append([i for i in range(len(a)) if a[i] == TILEname[8]])
###Construct Covnet
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=64, kernel_size=kernelsize1, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=128, kernel_size=kernelsize2, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernelsize3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout(0.5)
        self.fc1 = nn.Linear(192, 100)
        self.fc2 = nn.Linear(100, 2)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out =  self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        self.feature=out
        out = self.fc2(out)
        return torch.sigmoid(out)
model = ConvNet()
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
model=model.float()
model=model.apply(weight_init)
##Use GPU for computing
if torch.cuda.is_available():
    model.cuda()
### Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
Prediction=[]
y_standard=[]
prs=[]
aucs = []
mean_recall = np.linspace(0, 1, 100)
plt.figure(dpi=300,figsize=(15,12))
Eval_result=[{},{},{},{},{},{},{},{},{}]
torch.backends.cudnn.enabled=False
Feature_TILE=[]
Feature_M1=[]
for i in range(0, 9):
    X_train=torch.from_numpy(np.delete(Sequence,cv[i],0)).double()
    X_train=torch.transpose(X_train, 1, 2)
    X_train=X_train.to(torch.float32)
    X_train=X_train.cuda()
    M1=torch.from_numpy(Sequence[18421:]).double()
    M1=torch.transpose(M1, 1, 2)
    M1=M1.to(torch.float32)
    M1=M1.cuda()
    y_train=torch.from_numpy(np.delete(y,cv[i]))
    y_train=y_train.cuda()
    X_val=torch.from_numpy(Sequence[cv[i]])
    X_val=torch.transpose(X_val, 1, 2)
    X_val=X_val.to(torch.float32)
    X_val=X_val.cuda()
    y_val=torch.from_numpy(y[cv[i]])
    y_val=y_val.cuda()
    X_train1, X_test, y_train1, y_test = train_test_split(X_train, y_train, test_size = 0.2,random_state=42)
    train_dataset=TensorDataset(X_train, y_train)
    test_dataset=TensorDataset(X_val, y_val)
    M1_dataset=TensorDataset(M1)
    train1_dataset=TensorDataset(X_train1, y_train1)
    val_dataset=TensorDataset(X_test, y_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    M1_loader=DataLoader(dataset=M1_dataset, batch_size=batch_size, shuffle=False)
    train1_loader = DataLoader(dataset=train1_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    model=model.apply(weight_init)
   # Train the model
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs[i]):
        for j, (sequences, labels) in enumerate(train_loader):
            outputs = model(sequences)
            loss = criterion(outputs, labels.long())
            loss_list.append(loss.item())
            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)
            if ((j + 1) % 30 == 0)&(epoch%5==0):
                print('Epoch [{}/{}], Step [{}/{}],Gene {}, Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs[i], j + 1, total_step,i+1, loss.item(),
                              (correct / total) * 100))
                model.eval()
                prediction=[]
                pred=[]
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for sequences, labels in test_loader:
                        outputs = model(sequences)
                        prediction.extend(outputs.tolist())
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                pred=[a[1] for a in prediction]
                precision,recall,thresholds=metrics.precision_recall_curve(y_val.data.cpu(),pred)
                prs.append(np.interp(mean_recall, precision, recall))
                prc_auc=metrics.auc(recall,precision)
                aucs.append(prc_auc)
                print(prc_auc)
    model.eval()
    prediction=[]
    pred=[]
    feature_M1=[]
    with torch.no_grad():
        correct = 0
        total = 0
        for sequences, labels in test_loader:
            outputs = model(sequences)
            feature_TILE=model.feature
            prediction.extend(outputs.tolist())
            Feature_TILE.extend(feature_TILE.tolist())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        for sequences,  in M1_loader:
            outputs = model(sequences)
            feature_latent=model.feature
            feature_M1.extend(feature_latent.tolist())
        print('Test Accuracy of the model on the test Gene {} shRNAs: {} %'.format(i+1,(correct / total) * 100))
    pred=[a[1] for a in prediction]
    precision,recall,thresholds=metrics.precision_recall_curve(y_val.data.cpu(),pred)
    prs.append(np.interp(mean_recall, precision, recall))
    prc_auc=metrics.auc(recall,precision)
    aucs.append(prc_auc)
    plt.plot(recall,precision,lw=1,label='PRC Gene %d (area=%0.3f)' % (i+1, prc_auc))
    Prediction.extend(pred)
    print(prc_auc)
    Feature_M1.append(feature_M1)
    y_standard.extend(y_val.data.cpu())
precision,recall,thresholds=metrics.precision_recall_curve(y_standard,Prediction)
prc_auc=metrics.auc(recall,precision)
plt.plot(recall,precision,lw=3,color='red',label='PRC For All Gene (area=%0.3f)' % ( prc_auc))
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall',fontsize=20)
plt.ylabel('Precision',fontsize=20)
plt.title('Precision Recall Curve',fontsize=30)
plt.legend(loc="lower left",fontsize=20)
plt.show()  
###Plot ROC AUC
a=[]
b=Prediction
plt.figure(dpi=300,figsize=(15,12))
for i in range(0,9) :
    y_val=y[cv[i]]
    a.extend(y_val)
###Compute ROC curve and area the curve
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
plt.figure(dpi=300,figsize=(15,12))
ind=0
for i in range(0,9) :
    y_val=y[cv[i]]
    fpr,tpr,thresholds=metrics.roc_curve(y_val,Prediction[ind:(ind+len(cv[i]))],pos_label=1)
    ind=ind+len(cv[i])
    mean_tpr+=interp(mean_fpr,fpr,tpr)
    mean_tpr[0]=0.0
    roc_auc=metrics.auc(fpr,tpr)
    plt.plot(fpr,tpr,lw=1,label='ROC Gene %d (area=%0.3f)' % (i+1, roc_auc))
fpr,tpr,thresholds=metrics.roc_curve(a,b,pos_label=1)
roc_auc=metrics.auc(fpr,tpr)
plt.plot(fpr,tpr,lw=3,color='red',label='ROC For All Genes (area=%0.3f)' % (roc_auc))
plt.plot([0,1],[0,1],'--',color=(0.6,0.6,0.6),label='luck')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate',fontsize=20)
plt.ylabel('True Positive Rate',fontsize=20)
plt.title('Receiver operating characteristic ',fontsize=30)
plt.legend(loc="lower right",fontsize=20)
plt.show()  
torch.save(model,r'.\model.pt')
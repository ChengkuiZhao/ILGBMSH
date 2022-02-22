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
num_classes = 2
batch_size = 500
learning_rate = 0.001
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
#Sequence=np.array(df)
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
###构造rnn
input_size=4
output_size=2
hidden_dim=500
n_layers=2
class myRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(myRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bilstm = nn.LSTM(input_size,hidden_dim,bidirectional=True,batch_first=True)
        self.fc = nn.Linear(hidden_dim*2, output_size)
    def forward(self, x):
        batch_size = x.size(0)
        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)
        # Passing in the input and hidden state into the model and obtaining outputs
        print(x.shape)
        out,hidden= self.bilstm(x)
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.fc(out[:,-1,:])
        return out
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda()
        return hidden
model = myRNN(input_size, output_size, hidden_dim, n_layers)
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
if torch.cuda.is_available():
    model.cuda()
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
for i in range(0, 9):
    X_train=torch.from_numpy(np.delete(Sequence,cv[i],0)).double()
    X_train=X_train.to(torch.float32)
    X_train=X_train.cuda()
    y_train=torch.from_numpy(np.delete(y,cv[i]))
    y_train=y_train.cuda()
    X_val=torch.from_numpy(Sequence[cv[i]])
    X_val=X_val.to(torch.float32)
    X_val=X_val.cuda()
    y_val=torch.from_numpy(y[cv[i]])
    y_val=y_val.cuda()
    X_train1, X_test, y_train1, y_test = train_test_split(X_train, y_train, test_size = 0.2,random_state=42)
    train_dataset=TensorDataset(X_train, y_train)
    test_dataset=TensorDataset(X_val, y_val)
    train1_dataset=TensorDataset(X_train1, y_train1)
    val_dataset=TensorDataset(X_test, y_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    train1_loader = DataLoader(dataset=train1_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
   # Train the model
    model = myRNN(input_size, output_size, hidden_dim, n_layers)
    model=model.float()
    model=model.apply(weight_init)
    if torch.cuda.is_available():
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for j, (sequences, labels) in enumerate(train_loader):
            # Run the forward pass
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
                      .format(epoch + 1, num_epochs, j + 1, total_step,i+1, loss.item(),
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
    with torch.no_grad():
        correct = 0
        total = 0
        for sequences, labels in test_loader:
            outputs = model(sequences)
            prediction.extend(outputs.tolist())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model on the test Gene {} shRNAs: {} %'.format(i+1,(correct / total) * 100))
    pred=[a[1] for a in prediction]
    precision,recall,thresholds=metrics.precision_recall_curve(y_val.data.cpu(),pred)
    prs.append(np.interp(mean_recall, precision, recall))
    prc_auc=metrics.auc(recall,precision)
    aucs.append(prc_auc)
    plt.plot(recall,precision,lw=1,label='PRC Gene %d (area=%0.3f)' % (i+1, prc_auc))
    Prediction.extend(pred)
    print(prc_auc)
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

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import math
import scipy.sparse as sp
print("test")

writer = SummaryWriter()

train_set = pd.read_csv("data/final_format/train_set.csv",header=None).to_numpy()
train_label = pd.read_csv("data/final_format/train_label.csv",header=None).to_numpy()
test_set = pd.read_csv("data/final_format/test_set.csv",header=None).to_numpy()
test_label = pd.read_csv("data/final_format/test_label.csv",header=None).to_numpy()

#delet first row data
train_set = train_set[1:]
train_label = train_label[1:]
test_set = test_set[1:]
test_label = test_label[1:]

train_set = train_set.reshape((-1,64,64))
test_set = test_set.reshape((-1,64,64))

print(train_set.shape, train_label.shape, test_set.shape, test_label.shape)

#one hot encoding
def onehot(input):
    input_holder = np.zeros((1,4))
    for i in range(len(input)):
        temp = np.zeros((1,4))
        temp[0,int(input[i])] = 1
        input_holder = np.concatenate((input_holder,temp))

    input_holder = input_holder[1:]
    return input_holder

train_label = onehot(train_label)
test_label = onehot(test_label)
print(train_label.shape,test_label.shape)

#read adjacency matrix
adj = pd.read_csv("gcn_data/Adjacency_Matrix.csv",header=None).to_numpy()
adj_delta = adj + np.eye(64)
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
adj_delta = normalize(adj_delta)
adj_delta = Tensor(adj_delta).type(torch.float32)


#定义图卷积层
class GraphConvolution(nn.Module):

    
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
 
    def forward(self, input, adj):

        #特征变换
        support = torch.mm(input, self.weight)
        #邻居聚合
        output = torch.spmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output
    

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, fhid):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, fhid)
        self.d1 = nn.Dropout(p=0.5)     
        
        self.fc16 = nn.Linear(16, 4)
        self.fc32 = nn.Linear(32, 4)
    
    #定义前向计算，即各个图卷积层之间的计算逻辑
    def forward(self, x, adj):
        x = self.gc1(x, adj)
        output_1 = self.fc32(x)
        x = F.relu(x)
        x = self.d1(x)
        #第二层的输出
        x = self.gc2(x, adj)
        output_2 = self.fc16(x)
        result = torch.sum(output_1,0) + torch.sum(output_2,0)
        return result



# Hyper parameters
num_epochs = 3000
num_classes = 4
batch_size = 1
learning_rate = 1e-4



train_set_tensor = Tensor(train_set) 
train_label_tensor = Tensor(train_label).type(torch.LongTensor)

train_dataset = TensorDataset(train_set_tensor,train_label_tensor) 
train_loader = DataLoader(train_dataset, batch_size=batch_size) 

test_set_tensor = Tensor(test_set) 
test_label_tensor = Tensor(test_label).type(torch.LongTensor)

test_dataset = TensorDataset(test_set_tensor,test_label_tensor) 
test_loader = DataLoader(test_dataset, batch_size=batch_size) 

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = GCN(64,32,16).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3) 
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
milestones = [50,100,150,200,250]
milestones = [a * len(train_loader) for a in milestones]
scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.5)


# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    correct=0
    total=0
    running_loss = 0
    for i, (X, Y) in enumerate(train_loader):
        X = torch.squeeze(X)
        X = X.to(device)
        Y = Y.type(torch.DoubleTensor).to(device)
        adj_delta = adj_delta.to(device)

        # Forward pass
        outputs = model(X,adj_delta)
        outputs = torch.reshape(outputs, (1, 4))
        
        #print(outputs.shape,Y.shape)
        #print(outputs,Y)
        loss = criterion(outputs, Y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        #scheduler.step() 
        #print(scheduler.get_last_lr()[0])

        optimizer.step()
        scheduler.step() 
        #print(optimizer.param_groups[0]["lr"])

        _, predicted = outputs.max(1)
        _, tempY = Y.max(1)
        total += Y.size(0)
        if predicted == tempY:
            correct += 1
        running_loss += loss.item()
        accu=100.*correct/total
        train_loss = running_loss/(i+1)
        print ('Epoch [{}/{}], Step [{}/{}], Training Accuracy: {:.4f}%, Training Loss: {:.4f}%'.format(epoch+1, num_epochs, i+1, total_step, accu, train_loss))


        #writer.add_scalar(f'train/accuracy', accu, epoch)
        #writer.add_scalar(f'train/loss', train_loss, epoch)
        writer.add_scalars(f'train/accuracy_loss', {
            'accuracy': accu,
            'loss': train_loss,
        }, epoch)


# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for X, Y in test_loader:
        X = torch.squeeze(X)
        X = X.to(device)
        Y = Y.to(device)
        adj_delta = adj_delta.to(device)
        outputs = model(X,adj_delta)
        outputs = torch.reshape(outputs, (1, 4))
        _, predicted = torch.max(outputs.data, 1)
        _, tempY = Y.max(1)
        total += Y.size(0)
        if predicted == tempY:
            correct += 1

    print('Test Accuracy : {} %'.format(100 * correct / total))

# Save the model checkpoint
#torch.save(model.state_dict(), 'model.ckpt')
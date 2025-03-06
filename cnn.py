from torch import nn
import torch
from data import load_train_sections, split_dict
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from bayes_opt import BayesianOptimization
from Ensemble import Ensemble
import math
import pandas as pd
from base_model import BaseModel
from Sections import Sections

class Block(nn.Module):
    def __init__(self, start_layer, end_layer, kernal=3,dropout=0.1):
        super().__init__()
        self.start = torch.nn.Conv1d(start_layer, end_layer,1,padding='same')
        self.skip_start = start_layer != end_layer

        self.conv1 = torch.nn.Conv1d(end_layer, end_layer,kernal,padding='same')
        self.conv2 = torch.nn.Conv1d(end_layer, end_layer,kernal,padding='same')

        self.act = torch.nn.ReLU()

        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        if self.skip_start:
            x = self.start(x)
        
        temp = x

        x = self.conv1(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.dropout(x)

        return x + temp


# https://github.com/thuml/Time-Series-Library
class CNNRegress(Sections):
    def __init__(self, base_model, hidden_dim, change_rate, num_layers=2, dropout=0.1, kernal=3):
        super(CNNRegress, self).__init__()
        self.hidden_dim = hidden_dim
        self.change_rate = change_rate
        self.num_layers = num_layers  

        # with dimensionality hidden_dim.
        embedding_dim = 58

        self.downs = [Block(embedding_dim,hidden_dim, kernal, dropout)]+ [
            Block(hidden_dim + (i-1) * change_rate,hidden_dim + i * change_rate,kernal,dropout) 
            for i in range(1,num_layers)]
        self.middle = Block(hidden_dim + (num_layers-1) * change_rate, hidden_dim + (num_layers-1) * change_rate,kernal,dropout)
        
        self.ups = [Block(hidden_dim + i * change_rate,hidden_dim + (i-1) * change_rate, kernal,dropout) 
            for i in range(num_layers-1,0,-1)] + [
                Block(hidden_dim,hidden_dim, kernal, dropout)
            ]
        
        self.end = torch.nn.Conv1d(hidden_dim, 33,1,padding='same')

        
        
        self.dropout = nn.Dropout(dropout)
        # The linear layer that maps from hidden state space to tag space

        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()

        self.base_model = base_model

        self.upsample = torch.nn.Upsample(scale_factor=2,mode='linear', align_corners=True)

    def forward(self, x: torch.Tensor, store):
        base = self.base_model.predict(store,x)

        x = torch.transpose(x,0,1)
        # adding more stuff at the end just made it worse.
        temp = []
        for down in self.downs:
            x = down(x)
            temp.append(x)
            x = torch.max_pool1d(x,2)

        x = self.middle(x)

        
        for up in self.ups:
            x = torch.unsqueeze(x,0)
            x = self.upsample(x)
            x = torch.squeeze(x)

            part = temp.pop()
            x = x + part

            x = up(x)
        
        x = self.end(x)

        x = torch.transpose(x,0,1)

        x = self.softplus(x)
        x *= torch.transpose(base, 0, 1)

        return x
        
    
    def fit(self, xs: dict, ys: dict, loops=25):
        for store in xs.keys():
            
            if not isinstance(xs[store], torch.Tensor):
                xs[store] = torch.from_numpy(xs[store].values).to(torch.float32)
                
            if not isinstance(ys[store], torch.Tensor):
                ys[store] = torch.from_numpy(ys[store].values).to(torch.float32)
        
            
        
        optimizer = torch.optim.Adam(self.parameters())
        MSELoss = nn.MSELoss()
        for loop in tqdm(range(loops)):
            self.train()
            total_loss = 0
            count = 0
            for store in xs.keys():
                dataset = TensorDataset(xs[store], ys[store])
                loader = DataLoader(dataset, batch_size=256, shuffle=False,drop_last=True)
                
                for x, y in loader:
                    optimizer.zero_grad()

                    outputs = self(x,store)
                    mask = ~torch.isnan(y)
                    loss = MSELoss(outputs[mask],y[mask])

                    total_loss += loss.item()
                    count += len(x)

                    loss.backward()
                    optimizer.step()


    def predict(self,xs,x_before):
        for store in xs.keys():
            if not isinstance(xs[store], torch.Tensor):
                xs[store] = torch.from_numpy(xs[store].values).to(torch.float32)
            
            if not isinstance(x_before[store], torch.Tensor):
                x_before[store] = torch.from_numpy(x_before[store].values).to(torch.float32)

        outs = dict()
    
        with torch.no_grad():
            self.eval()
            for store in xs.keys():
                N = len(xs[store])

                extra = 256 - (N % 256)

                temp = torch.cat([x_before[store][-extra:],xs[store]])
                dataset = TensorDataset(temp)
                loader = DataLoader(dataset, batch_size=256, shuffle=False,drop_last=True)
                
                temp = []
                for x in loader:
                    x = x[0]
                    outputs = self(x,store)
                    temp.append(outputs)
                temp = torch.concat(temp)

                outs[store] = temp[extra:]
        return outs
                

def getParamters():
    x_train, y_train, x_test, y_test = load_train_sections()
    print('loaded')
    print(x_train[1].columns.get_loc('day_count'))

    param_space = {
        'hidden_dim': (50, 120),
        'change_rate': (1, 10),
        'dropout': (0.0, 0.5),
        'num_layers': (1, 4),
        'kernal': (3,7),
    }

    x_0, x_1 = split_dict(x_train)
    y_0, y_1 = split_dict(y_train)

    base_model = BaseModel(x_0,y_0,2)

    def objective(hidden_dim=40, change_rate=1, num_layers=2, dropout=0.1, kernal=3):
        change_rate = int(round(change_rate))
        hidden_dim = int(round(hidden_dim))
        num_layers = int(round(num_layers))
        kernal = int(round(kernal))
        if kernal % 2 == 0:
            kernal -= 1


        
        model = CNNRegress(base_model,hidden_dim,change_rate,num_layers,dropout,kernal)
        model.fit(x_0,y_0)

        temp = model.score(x_1,y_1,x_0)
        print(temp)
        return temp

    model = BayesianOptimization(
        f=objective,
        pbounds=param_space,
    )

    model.maximize()

    # {'target': 0.8887208583296505, 'params': {'change_rate': 9.622603430952681, 'dropout': 0.3246723642615743, 'hidden_dim': 93.25033043554573, 'kernal': 4.1879906541091, 'num_layers': 1.504282461338219}}
    print(model.max)
    return model.max

def makeModel():
    x_train, y_train, x_test, y_test = load_train_sections()

    base_model = BaseModel(x_train,y_train,2)
    model = CNNRegress(base_model,70,2,0.3067193001160901,0.012635622343381815)
    model.fit(x_train,y_train)
    
    # 0.9132228131467848
    print(model.score(x_test,y_test,x_train))
    return model

    

if __name__ == '__main__':
    getParamters()
    

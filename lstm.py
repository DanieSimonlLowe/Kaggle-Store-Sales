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
import numpy as np

# https://github.com/thuml/Time-Series-Library
class LSTMRegress(Sections):
    def __init__(self, hidden_dim, num_layers=2, dropout_rate=0.1, l2=0.001):
        super(LSTMRegress, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers  

        # with dimensionality hidden_dim.
        embedding_dim = 58 + 33

        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        # The linear layer that maps from hidden state space to tag space

        self.hidden2out = nn.Linear(hidden_dim, 33)
        self.relu = nn.ReLU()

        self.l2 = l2

    def forward(self, x: torch.Tensor, store, h0=None, c0=None):
        x = self.dropout(x)
        if h0 is None or c0 is None:
            lstm_out, (hn, cn) = self.lstm(x)
        else:
            lstm_out, (hn, cn) = self.lstm(x, (h0, c0))

        # adding more stuff at the end just made it worse.
        out = self.dropout(lstm_out)
        out = self.hidden2out(out)

        return out, hn, cn
        
    
    def fit(self, xs: dict, ys: dict, loops=25):
        
        
            
        
        optimizer = torch.optim.Adam(self.parameters(), weight_decay=self.l2)
        MSELoss = nn.MSELoss()
        for loop in tqdm(range(loops)):
            self.train()
            total_loss = 0
            for store in xs.keys():
                dataset = TensorDataset(xs[store], ys[store])
                loader = DataLoader(dataset, batch_size=256, shuffle=False)
                
                h0, c0 = None, None
                for x, y in loader:
                    optimizer.zero_grad()
                    
                    outputs, h0, c0 = self(x,store,h0,c0)
                    

                    mask = ~torch.isnan(y)
                    loss = MSELoss(outputs[mask],y[mask])
                    if not torch.isnan(loss):
                        total_loss += loss.item()
                        #print(outputs)
                        
                        loss.backward()
                        optimizer.step()

                    h0 = h0.detach()
                    c0 = c0.detach()
            print('')
            print(total_loss)


    def predict(self,xs,temp):
        x_before, y_before = temp

        outs = dict()
    
        with torch.no_grad():
            self.eval()
            for store in xs.keys():
                y_pred = y_before[store][-1]

                y_pred_true = False
                if store in x_before:
                    y_pred_true, h0, c0 = self(x_before[store],store)
                    y_pred_true = y_pred_true[-1]
                else:
                    h0, c0 = None, None
                x = xs[store]


                y_pred = torch.where(torch.isnan(y_pred), y_pred_true, y_pred)

                temp = []
                for part in x:
                    part = torch.concat([part,y_pred])
                    part = torch.unsqueeze(part,0)
                    y_pred, h0, c0 = self(part, store, h0, c0)
                    y_pred = torch.squeeze(y_pred)
                    if torch.isnan(y_pred).any():
                        print(y_pred)
                    temp.append(y_pred)

                outs[store] = torch.from_numpy(np.array(temp)).to(torch.float32)
        return outs


def  getParamters():
    x_train, y_train, x_test, y_test = load_train_sections()
    print('loaded')

    for store in y_train.keys():
        print(torch.isnan(torch.from_numpy(y_train[store].values)).any())

    param_space = {
        'hidden_dim': (50, 180),
        'num_layers': (1, 3),
        'dropout_rate': (0.0, 0.5),
        #'l2': (0, 0.001),
    }

    x_0, x_1 = split_dict(x_train)
    y_0, y_1 = split_dict(y_train)

    base_model = BaseModel(x_0,y_0,1)

    base_model.transform(x_0,y_0)
    base_model.transform_y(x_1,y_1)

    def objective(hidden_dim=70, num_layers=2, dropout_rate=0.1, l2=0.0):
        hidden_dim = int(round(hidden_dim))
        num_layers = int(round(num_layers))

        
        model = LSTMRegress(hidden_dim,num_layers,dropout_rate,l2)
        model.fit(x_0,y_0)

        return model.score(x_1,y_1,(x_0,y_0))

    model = BayesianOptimization(
        f=objective,
        pbounds=param_space,
    )

    model.maximize()

    # {'target': 0.9181133072782891, 'params': {'dropout_rate': 0.3067193001160901, 'hidden_dim': 70.13865827406954, 'l2': 0.012635622343381815, 'num_layers': 2.0385343002566065}}
    print(model.max)
    return model.max

def makeModel():
    x_train, y_train, x_test, y_test = load_train_sections()

    base_model = BaseModel(x_train,y_train,2)
    model = LSTMRegress(base_model,70,2,0.3067193001160901,0.012635622343381815)
    model.fit(x_train,y_train)
    
    # 0.9132228131467848
    print(model.score(x_test,y_test,x_train))
    return model

def makeEnsemble():
    x_train, y_train, x_test, y_test = load_train_sections()
    base_model = BaseModel(x_train,y_train,2)

    x_0, x_1 = split_dict(x_train)
    y_0, y_1 = split_dict(y_train)
    out = []
    for i in range(10):
        print(i)
        model = LSTMRegress(base_model,70,2,0.3067193001160901,0.012635622343381815)
        model.fit(x_0,y_0)
        out.append(model)
    
    model = Ensemble(out)
    model.fit(x_1,y_1,x_0)

    print(model.score(x_test,y_test,x_train))
    

    

if __name__ == '__main__':
    getParamters()
    

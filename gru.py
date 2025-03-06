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

# https://github.com/thuml/Time-Series-Library
class GRURegress(Sections):
    def __init__(self, base_model, hidden_dim, num_layers=2, dropout_rate=0.1, lr=0.001):
        super(GRURegress, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers  

        # with dimensionality hidden_dim.
        embedding_dim = 58

        self.gru = nn.GRU(embedding_dim, hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        # The linear layer that maps from hidden state space to tag space

        self.hidden2out = nn.Linear(hidden_dim, 33)
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()

        self.lr = lr
        self.base_model = base_model

    def forward(self, x: torch.Tensor, store, h0=None):
        
        if h0 is None:
            out, hn = self.gru(x)
        else:
            out, hn = self.gru(x, h0)

        # adding more stuff at the end just made it worse.
        out = self.dropout(out)
        out = self.hidden2out(out)
        out = self.softplus(out)

        temp = self.base_model.predict(store,x)
        out *= torch.transpose(temp, 0, 1)

        return out, hn
            

    
    def fit(self, xs: dict, ys: dict, loops=25):
        for store in xs.keys():
            
            if not isinstance(xs[store], torch.Tensor):
                xs[store] = torch.from_numpy(xs[store].values).to(torch.float32)
                
            if not isinstance(ys[store], torch.Tensor):
                ys[store] = torch.from_numpy(ys[store].values).to(torch.float32)
        
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        MSELoss = nn.MSELoss()
        for loop in tqdm(range(loops)):
            self.train()
            total_loss = 0
            count = 0
            for store in xs.keys():
                dataset = TensorDataset(xs[store], ys[store])
                loader = DataLoader(dataset, batch_size=256, shuffle=False)
                
                h0 = None
                for x, y in loader:
                    optimizer.zero_grad()

                    outputs, h0 = self(x,store,h0)
                    mask = ~torch.isnan(y)
                    loss = MSELoss(outputs[mask],y[mask])

                    total_loss += loss.item()
                    count += len(x)

                    loss.backward()
                    optimizer.step()

                    h0 = h0.detach()


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
                if store in x_before:
                    _, h0 = self(x_before[store],store)
                else:
                    h0 = None

                x = xs[store]
                
                y_pred, _ = self(x, store, h0)

                outs[store] = y_pred
        return outs
                

def getParamters():
    x_train, y_train, x_test, y_test = load_train_sections()
    print('loaded')
    print(x_train[1].columns.get_loc('day_count'))

    param_space = {
        'hidden_dim': (50, 110),
        'num_layers': (1, 4),
        'dropout_rate': (0.0, 0.5),
        'lr': (0, 0.1),
        'power': (1,4),
    }

    

    def objective(hidden_dim=70, num_layers=2, dropout_rate=0.1, lr=0.001,power=2):
        power = int(round(power))
        hidden_dim = int(round(hidden_dim))
        num_layers = int(round(num_layers))

        x_0, x_1 = split_dict(x_train)
        y_0, y_1 = split_dict(y_train)

        base_model = BaseModel(x_0,y_0,power)
        
        model = GRURegress(base_model,hidden_dim,num_layers,dropout_rate,l2)
        model.fit(x_0,y_0)

        temp = model.score(x_1,y_1,x_0)
        print(temp)
        return temp

    model = BayesianOptimization(
        f=objective,
        pbounds=param_space,
    )

    model.maximize()

    # {'target': 0.9181133072782891, 'params': {'dropout_rate': 0.3067193001160901, 'hidden_dim': 70.13865827406954, 'lr': 0.012635622343381815, 'num_layers': 2.0385343002566065}}
    print(model.max)
    return model.max

def makeModel():
    x_train, y_train, x_test, y_test = load_train_sections()

    base_model = BaseModel(x_train,y_train,2)
    model = GRURegress(base_model,70,2,0.3067193001160901,0.012635622343381815)
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
        model = GRURegress(base_model,70,2,0.3067193001160901,0.012635622343381815)
        model.fit(x_0,y_0)
        out.append(model)
    
    model = Ensemble(out)
    model.fit(x_1,y_1,x_0)

    print(model.score(x_test,y_test,x_train))
    

    

if __name__ == '__main__':
    # Very Very slow so don't use it
    getParamters()
    

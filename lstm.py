from torch import nn
import torch
from data import load_train_sections, split_dict
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from bayes_opt import BayesianOptimization
from sklearn.linear_model import LinearRegression
import math
import pandas as pd
from base_model import BaseModel

class LSTMRegress(nn.Module):
    def __init__(self, base_model, hidden_dim, num_layers=2, dropout_rate=0.1, l2=0.001):
        super(LSTMRegress, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers  

        # with dimensionality hidden_dim.
        embedding_dim = 58

        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2out = nn.Linear(hidden_dim, 33)
        self.softplus = nn.Softplus()

        self.l2 = l2
        self.base_model = base_model

    def forward(self, x: torch.Tensor, store, h0=None, c0=None):
        
        if h0 is None or c0 is None:
            lstm_out, (hn, cn) = self.lstm(x)
        else:
            lstm_out, (hn, cn) = self.lstm(x, (h0, c0))

        out = self.dropout(lstm_out)
        out = self.hidden2out(out)
        out = self.softplus(out)

        temp = self.base_model.predict(store,x)
        out *= torch.transpose(temp, 0, 1)

        return out, hn, cn
            

    
    def fit(self, xs: dict, ys: dict, loops=25):
        for store in xs.keys():
            
            if not isinstance(xs[store], torch.Tensor):
                xs[store] = torch.from_numpy(xs[store].values).to(torch.float32)
                
            if not isinstance(ys[store], torch.Tensor):
                ys[store] = torch.from_numpy(ys[store].values).to(torch.float32)
        
            
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.l2)
        MSELoss = nn.MSELoss()
        for loop in tqdm(range(loops)):
            self.train()
            total_loss = 0
            count = 0
            for store in xs.keys():
                #x = xs[store]
                #y = ys[store]
                dataset = TensorDataset(xs[store], ys[store])
                loader = DataLoader(dataset, batch_size=256, shuffle=False)
                
                h0, c0 = None, None
                for x, y in loader:
                    optimizer.zero_grad()

                    outputs, h0, c0 = self(x,store,h0,c0)
                    mask = ~torch.isnan(y)
                    loss = MSELoss(outputs[mask],y[mask])

                    total_loss += loss.item()
                    count += len(x)

                    loss.backward()
                    optimizer.step()

                    h0 = h0.detach()
                    c0 = c0.detach()     

    def score(self, xs: dict, ys: dict, x_before: dict):
        for store in xs.keys():
            if not isinstance(xs[store], torch.Tensor):
                xs[store] = torch.from_numpy(xs[store].values).to(torch.float32)
            if not isinstance(ys[store], torch.Tensor):
                ys[store] = torch.from_numpy(ys[store].values).to(torch.float32)
            
            if not isinstance(x_before[store], torch.Tensor):
                x_before[store] = torch.from_numpy(x_before[store].values)

        with torch.no_grad():
            self.eval()
            ss_res = 0
            ss_tot = 0
            for store in xs.keys():
                if store in x_before:
                    _, h0, c0 = self(x_before[store],store)
                else:
                    h0, c0 = None, None

                x = xs[store]
                y = ys[store]
                
                y_pred, _, _ = self(x, store, h0, c0)

                y = y.to(y_pred.device)
                mask = ~torch.isnan(y)

                ss_res += torch.sum((y[mask] - y_pred[mask]) ** 2).item()
                ss_tot += torch.sum((y[mask] - torch.mean(y[mask])) ** 2).item()
                
        temp = 1 - ss_res / (ss_tot + 1e-8)

        if math.isnan(temp):
            return -100
        else:
            return temp



def getParamters():
    x_train, y_train, x_test, y_test = load_train_sections()
    print('loaded')
    print(x_train[1].columns.get_loc('day_count'))

    param_space = {
        'hidden_dim': (20, 200),
        'num_layers': (1, 4),
        'dropout_rate': (0.0, 0.5),
        'l2': (0, 5e-2)
    }

    base_model = BaseModel(x_train,y_train)

    def objective(hidden_dim=10, num_layers=2, dropout_rate=0.1, l2=0.001):
        hidden_dim = int(round(hidden_dim))
        num_layers = int(round(num_layers))

        x_0, x_1 = split_dict(x_train)
        y_0, y_1 = split_dict(y_train)
        model = LSTMRegress(base_model,hidden_dim,num_layers,dropout_rate,l2)
        model.fit(x_0,y_0)

        temp = model.score(x_1,y_1,x_0)
        print(temp)
        return temp

    model = BayesianOptimization(
        f=objective,
        pbounds=param_space,
    )

    model.maximize(n_iter=30)

    # {'target': 0.9181133072782891, 'params': {'dropout_rate': 0.3067193001160901, 'hidden_dim': 70.13865827406954, 'l2': 0.012635622343381815, 'num_layers': 2.0385343002566065}}
    print(model.max)
    return model.max

def makeModel():
    x_train, y_train, x_test, y_test = load_train_sections()

    # model = LSTMRegress(58,hidden_dim,33,num_layers,dropout_rate,l2)
    # model.fit(x_train, y_train)

    # print(model.score(x_test,y_test))

    

if __name__ == '__main__':
    getParamters()
    

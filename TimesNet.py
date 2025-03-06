import torch
from torch import nn
import torch.nn.functional as F
from embed import DataEmbedding
from tqdm import tqdm
from Sections import Sections
from data import load_train_sections, split_dict
from base_model import BaseModel
from torch.utils.data import TensorDataset, DataLoader
from bayes_opt import BayesianOptimization
from  random import randint

# from https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

class TimesBlock(nn.Module):
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ff, num_kernels):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff,
                               num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model,
                               num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res

class TimesNet(Sections):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, base_model, top_k, d_ff, num_kernels, e_layers, dropout, d_model,
                outputdim=33, enc_in=58, seq_len=256, label_len=256, pred_len=256, embed='fix', freq='f'):
        super(TimesNet, self).__init__()
        #self.configs = configs
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.model = nn.ModuleList([TimesBlock(seq_len, pred_len, top_k, d_model, d_ff, num_kernels)
                                    for _ in range(e_layers)])
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq,
                                           dropout)
        self.layer = e_layers
        self.layer_norm = nn.LayerNorm(d_model)

        self.base_model = base_model

        #self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(d_model, outputdim, bias=True)
        
        self.softplus = nn.Softplus()

    def forward(self, x, store):

        x_enc = torch.unsqueeze(x,0)
        # embedding
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-6)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        #enc_out = self.predict_linear(enc_out)

        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        
        dec_out = self.projection(enc_out)
        dec_out = torch.squeeze(dec_out)

        temp = self.base_model.predict(store,x)
        temp = torch.transpose(temp, 0, 1)
        dec_out = temp * dec_out
        
        return dec_out
    
    def fit(self,xs,ys, loops=25):
        for store in xs.keys():
            
            if not isinstance(xs[store], torch.Tensor):
                xs[store] = torch.from_numpy(xs[store].values).to(torch.float32)
                
            if not isinstance(ys[store], torch.Tensor):
                ys[store] = torch.from_numpy(ys[store].values).to(torch.float32)
        
        
        optimizer = torch.optim.Adam(self.parameters())
        MSELoss = nn.MSELoss()
        for loop in tqdm(range(loops)):
            self.train()
            extra = randint(0,len(xs[store]) % 512)
            dataset = TensorDataset(xs[store][extra:], ys[store][extra:])
            loader = DataLoader(dataset, batch_size=512, shuffle=False,drop_last=True)
                
            for x, y in loader:
                optimizer.zero_grad()
                mask = ~torch.isnan(y)

                outputs = self(x,store)
                
                loss = MSELoss(outputs[mask],y[mask])

                loss.backward()
                optimizer.step()


    def predict(self,xs,x_before):
        for store in xs.keys():
            if not isinstance(xs[store], torch.Tensor):
                xs[store] = torch.from_numpy(xs[store].values).to(torch.float32)
            
        outs = dict()

        
    
        with torch.no_grad():
            self.eval()
            for store in xs.keys():
                N = len(xs[store])

                extra = 512 - (N % 512)

                temp = torch.cat([x_before[store][-extra:],xs[store]])
                dataset = TensorDataset(temp)
                loader = DataLoader(dataset, batch_size=512, shuffle=False)
                
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
        'top_k': (1, 10),
        'd_ff': (34, 100),
        'dropout': (0.0, 0.5),
        'num_kernels': (1, 3),
        'e_layers': (1,4),
        'd_model': (34,150),
    }

    x_0, x_1 = split_dict(x_train)
    y_0, y_1 = split_dict(y_train)

    base_model = BaseModel(x_0,y_0,2)
    def objective(top_k, d_ff, num_kernels, e_layers, dropout, d_model):
        top_k = int(round(top_k))
        d_ff = int(round(d_ff))
        e_layers = int(round(e_layers))
        d_model = int(round(d_model))
        num_kernels = int(round(num_kernels))

        d_model -= d_model % 2
        
        model = TimesNet(base_model,top_k, d_ff, num_kernels, e_layers, dropout, d_model)
        model.fit(x_0,y_0)

        temp = model.score(x_1,y_1,x_0)
        print(temp)
        return temp

    model = BayesianOptimization(
        f=objective,
        pbounds=param_space,
    )

    model.maximize()

    # {'target': 0.8490333782917848, 'params': {'d_ff': 39.2673203124063, 'd_model': 104.03914197430309, 'dropout': 0.3181695825612751, 'e_layers': 3.352232955634117, 'num_kernels': 1.2498459288383374, 'top_k': 4.526925042874494}}
    print(model.max)
    return model.max

if __name__ == '__main__':
    getParamters()
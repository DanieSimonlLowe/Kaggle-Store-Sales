from torch import nn
import torch
import math

class Sections(nn.Module):
    def __init__(self):
        super(Sections, self).__init__()
    
    def score(self, xs: dict, ys: dict, x_before):
        for store in xs.keys():
            if not isinstance(xs[store], torch.Tensor):
                xs[store] = torch.from_numpy(xs[store].values).to(torch.float32)
            if not isinstance(ys[store], torch.Tensor):
                ys[store] = torch.from_numpy(ys[store].values).to(torch.float32)
            

        y_preds = self.predict(xs,x_before)

        with torch.no_grad():
            self.eval()
            ss_res = 0
            ss_tot = 0
            for store in xs.keys():
                y_pred = y_preds[store]
                y = ys[store]
                y = y.to(y_pred.device)
                mask = ~torch.isnan(y)

                ss_res += torch.sum((y[mask] - y_pred[mask]) ** 2).item()
                ss_tot += torch.sum((y[mask] - torch.mean(y[mask])) ** 2).item()
                    
        temp = 1 - ss_res / (ss_tot + 1e-8)

        if math.isnan(temp):
            return -100
        else:
            return temp

    def loss(self,xs:dict, ys:dict, x_before):
        for store in xs.keys():
            if not isinstance(xs[store], torch.Tensor):
                xs[store] = torch.from_numpy(xs[store].values).to(torch.float32)
            if not isinstance(ys[store], torch.Tensor):
                ys[store] = torch.from_numpy(ys[store].values).to(torch.float32)
            

        y_preds = self.predict(xs,x_before)

        with torch.no_grad():
            self.eval()
            loss = 0
            for store in xs.keys():
                y = ys[store]
                
                y_pred = y_preds[store]

                y = y.to(y_pred.device)
                mask = ~torch.isnan(y)
                loss += torch.sum((y[mask] - y_pred[mask]) ** 2).item()
            return loss
    
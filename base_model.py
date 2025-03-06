from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
import torch
import warnings
from sklearn.metrics import mean_squared_error
from math import sqrt

class BaseModel():

    def __init__(self, x, y, power=1):
        self.models = {}
        self.max_error = {}

        class DummyModel:
            def predict(self, X):
                return np.zeros((X.shape[0],))

        self.power = power
        for store in x.keys():
            temp = []
            temp2 = []
            for i in range(33):
                # Concatenate the 'day_count' column from x with the i-th column from y
                data = pd.concat([x[store][['day_count']], y[store].iloc[:, i]], axis=1)
                # Drop rows where the i-th column of y has NaN values
                data = data.dropna(subset=[y[store].columns[i]])

                if len(data) <= 0:
                    # If no data exists, use the dummy model that always predicts 0
                    dummy_model = DummyModel()
                    temp.append(dummy_model)
                    temp2.append(1)  # mse is set to 1 as specified
                    continue

                # Extract the cleaned 'day_count' and the i-th column of y
                x_cleaned = data[['day_count']]
                y_cleaned = data[y[store].columns[i]]

                # Create and train the LinearRegression model
                model = make_pipeline(
                    PolynomialFeatures(degree=self.power, include_bias=False),
                    LinearRegression()
                )
                model.fit(x_cleaned, y_cleaned)

                predictions = model.predict(x_cleaned)
                error = y_cleaned - predictions
                mse = max(np.max(np.abs(error)), 1e-6)
                #print(mse)

                
                # Append the trained model to the list
                temp.append(model)
                temp2.append(mse)
            self.models[store] = temp
            self.max_error[store] = torch.unsqueeze(torch.Tensor(temp2),0)
    
    def predict(self, store, x):
        warnings.filterwarnings("ignore", category=UserWarning)
        predictions = []
        for i in range(33):
            # Extract the 'day_count' column for prediction
            temp = x[:,39].reshape(-1, 1)
            # Predict using the i-th model and append the result
            predictions.append(self.models[store][i].predict(temp))
        return torch.from_numpy(np.array(predictions)).to(torch.float32)

    def transform_y(self, xs, ys):
        for store in ys.keys():
            if not isinstance(xs[store], torch.Tensor):
                xs[store] = torch.from_numpy(xs[store].values).to(torch.float32)

            predictions = torch.transpose(self.predict(store,xs[store]),0,1)

            if not isinstance(ys[store], torch.Tensor):
                ys[store] = torch.from_numpy(ys[store].values).to(torch.float32)
            
            nan_mask = torch.isnan(ys[store])
            diff = ys[store] - predictions
            normalized_diff = diff / self.max_error[store]

            print(nan_mask.any())
            ys[store] = torch.where(nan_mask, torch.nan, normalized_diff)


    
    
    def transform(self,xs,ys):
        self.transform_y(xs,ys)
        
        for store in ys.keys():
            if not isinstance(xs[store], torch.Tensor):
                xs[store] = torch.from_numpy(xs[store].values).to(torch.float32)

            temp = torch.roll(ys[store],1)
            temp = torch.nan_to_num(temp)
            xs[store] = torch.cat([xs[store],temp],1)

            xs[store] = xs[store][1:]
            ys[store] = ys[store][1:] - ys[store][:-1]


# if __name__ == '__main__':
#     create_base_model()
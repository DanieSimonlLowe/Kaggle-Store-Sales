from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import torch
import warnings

class BaseModel():

    def __init__(self, x, y):
        self.models = {}
        for store in x.keys():
            temp = []
            for i in range(33):
                # Concatenate the 'day_count' column from x with the i-th column from y
                data = pd.concat([x[store][['day_count']], y[store].iloc[:, i]], axis=1)
                # Drop rows where the i-th column of y has NaN values
                data = data.dropna(subset=[y[store].columns[i]])

                # Extract the cleaned 'day_count' and the i-th column of y
                x_cleaned = data[['day_count']]
                y_cleaned = data[y[store].columns[i]]

                # Create and train the LinearRegression model
                model = LinearRegression()
                model.fit(x_cleaned, y_cleaned)
                
                # Append the trained model to the list
                temp.append(model)
            self.models[store] = temp
    
    def predict(self, store, x):
        warnings.filterwarnings("ignore", category=UserWarning)
        predictions = []
        for i in range(33):
            # Extract the 'day_count' column for prediction
            temp = x[:,39].reshape(-1, 1)
            # Predict using the i-th model and append the result
            predictions.append(self.models[store][i].predict(temp))
        return torch.from_numpy(np.array(predictions)).to(torch.float32)


# if __name__ == '__main__':
#     create_base_model()
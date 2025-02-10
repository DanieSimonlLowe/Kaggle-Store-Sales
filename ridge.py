from data import loadTrain
from sklearn.linear_model import BayesianRidge

def makeModel():
    x_train, x_test, y_train, y_test = loadTrain()
    print('loaded')
    model = BayesianRidge()
    model.fit(x_train,y_train)
    # 0.6223929798974587
    print(model.score(x_test, y_test))
    
    return model

if (__name__ == '__main__'):
    #getHistagram()
    makeModel()
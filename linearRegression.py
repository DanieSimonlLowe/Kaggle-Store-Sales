from data import load_train_sections_combined
from sklearn.linear_model import LinearRegression

def makeModel():
    x_train, x_test, y_train, y_test = load_train_sections_combined()
    print('loaded')
    print(x_train.shape)
    print(y_train.shape)
    model = LinearRegression()
    model.fit(x_train,y_train)
    # 0.33383123542222565
    print(model.score(x_test, y_test))
    
    return model

if (__name__ == '__main__'):
    #getHistagram()
    makeModel()
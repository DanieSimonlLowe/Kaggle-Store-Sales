from Sections import Sections

class Ensemble(Sections):
    def __init__(self, models):
        self.models = models
    

    def fit(self, xs, ys, params = None):
        weights = []
        total = 0
        for model in self.models:
            loss = model.loss(xs,ys,params)
            weights.append(loss)
            total += loss
        
        for i in range(len(weights)):
            weights[i] = total / weights[i]
        

    def predict(self,xs,params):
        return sum(self.models[i].predict(xs,params) for i in range(len(self.models)))


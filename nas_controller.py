import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import uniform
from scipy.optimize import minimize
import random

class SurrogateModel: 
    def __init__(self):
        self.X = []
        self.y = []
        self.model = RandomForestRegressor()

    def encode_arch(self, arch_dict):
         # Converts architecture dict to a fixed-length numeric vector
         return [
            arch_dict["num_layers"],
            arch_dict["hidden_dim"],
            {"relu": 0, "tanh": 1}[arch_dict["activation"]],
        ]
    
    def update(self, arch_dict, performance_score):
        x = self.encode_arch(arch_dict)
        self.X.append(x)
        self.y.append(performance_score)
        self.model.fit(self.X, self.y)
    
    def predict(self, arch_dict):
        x = np.array(self.encode_arch(arch_dict)).reshape(1, -1)
        return self.model.predict(x)[0]
    
    def suggest(self, n_candidates=10, workload_meta=None):
        suggestions = []
        for _ in range(n_candidates):
            arch = {
                "num_layers": random.choice([1, 2, 3]),
                "hidden_dim": random.choice([16, 32, 64, 128]),
                "activation": random.choice(["relu", "tanh"]),
            }
            predicted_perf = self.predict(arch, workload_meta) if self.X else 0
            suggestions.append((arch, predicted_perf))
        suggestions.sort(key=lambda x: -x[1])
        return suggestions[:3]  # return top 3

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
            {"relu": 0, "tanh": 1, "leaky_relu": 2, "elu": 3, "swish": 4}[arch_dict["activation"]],
            {"linear": 0, "dropout": 1, "batch_norm": 2}[arch_dict.get("regularization", "linear")],
            arch_dict.get("dropout_rate", 0.0),
            arch_dict.get("learning_rate", 0.001),
            {"adam": 0, "sgd": 1, "rmsprop": 2}[arch_dict.get("optimizer", "adam")],
            arch_dict.get("batch_size", 32),
            {"mse": 0, "mae": 1, "huber": 2}[arch_dict.get("loss_function", "mse")],
            arch_dict.get("attention_heads", 0),  # 0 means no attention
            arch_dict.get("residual_connections", False),
            arch_dict.get("layer_normalization", False),
        ]
    
    def update(self, arch_dict, performance_score):
        x = self.encode_arch(arch_dict)
        self.X.append(x)
        self.y.append(performance_score)
        self.model.fit(self.X, self.y)
    
    def predict(self, arch_dict):
        x = np.array(self.encode_arch(arch_dict)).reshape(1, -1)
        return self.model.predict(x)[0]
    
    def suggest(self, n_candidates=20, workload_meta=None):
        suggestions = []
        for _ in range(n_candidates):
            arch = {
                "num_layers": random.choice([1, 2, 3, 4, 5, 6]),
                "hidden_dim": random.choice([16, 32, 64, 128, 256, 512]),
                "activation": random.choice(["relu", "tanh", "leaky_relu", "elu", "swish"]),
                "regularization": random.choice(["linear", "dropout", "batch_norm"]),
                "dropout_rate": random.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
                "learning_rate": random.choice([0.0001, 0.001, 0.01, 0.1]),
                "optimizer": random.choice(["adam", "sgd", "rmsprop"]),
                "batch_size": random.choice([16, 32, 64, 128]),
                "loss_function": random.choice(["mse", "mae", "huber"]),
                "attention_heads": random.choice([0, 1, 2, 4, 8]),  # 0 means no attention
                "residual_connections": random.choice([True, False]),
                "layer_normalization": random.choice([True, False]),
            }
            predicted_perf = self.predict(arch) if self.X else 0
            suggestions.append((arch, predicted_perf))
        suggestions.sort(key=lambda x: -x[1])
        return suggestions[:5]  # return top 5 instead of 3

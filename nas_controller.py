import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import random
import joblib
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class EnsembleSurrogateModel:
    def __init__(self):
        self.X = []
        self.y = []
        self.scaler = StandardScaler()
        
        # Ensemble of models
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'neural_net': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
            'gaussian_process': GaussianProcessRegressor(
                kernel=C(1.0) * RBF([1.0] * 13), 
                random_state=42,
                n_restarts_optimizer=5
            )
        }
        
        self.model_weights = {
            'random_forest': 0.3,
            'gradient_boosting': 0.3,
            'neural_net': 0.2,
            'gaussian_process': 0.2
        }
        
        self.trained = False
        self.uncertainty_history = []
        self.exploration_scores = []

    def encode_arch(self, arch_dict):
        """Enhanced encoding with workload and hardware features"""
        # Architecture features
        arch_features = [
            arch_dict["num_layers"],
            arch_dict["hidden_dim"],
            {"relu": 0, "tanh": 1, "leaky_relu": 2, "elu": 3, "swish": 4}[arch_dict["activation"]],
            {"linear": 0, "dropout": 1, "batch_norm": 2}[arch_dict.get("regularization", "linear")],
            arch_dict.get("dropout_rate", 0.0),
            arch_dict.get("learning_rate", 0.001),
            {"adam": 0, "sgd": 1, "rmsprop": 2}[arch_dict.get("optimizer", "adam")],
            arch_dict.get("batch_size", 32),
            {"mse": 0, "mae": 1, "huber": 2}[arch_dict.get("loss_function", "mse")],
            arch_dict.get("attention_heads", 0),
            int(arch_dict.get("residual_connections", False)),
            int(arch_dict.get("layer_normalization", False)),
        ]
        
        # Workload meta-features (if available)
        workload_features = [
            arch_dict.get("avg_task_duration", 3.0),
            arch_dict.get("task_duration_variance", 1.0),
            arch_dict.get("memory_intensity", 0.5),
            arch_dict.get("io_intensity", 0.3),
            arch_dict.get("dependency_density", 0.2),
            arch_dict.get("priority_variance", 0.5),
        ]
        
        # Hardware features
        hardware_features = [
            arch_dict.get("num_cores", 4),
            arch_dict.get("core_heterogeneity", 0.5),
            arch_dict.get("memory_bandwidth", 25.6),
            arch_dict.get("cache_size", 8),
        ]
        
        return arch_features + workload_features + hardware_features

    def update(self, arch_dict, performance_score, workload_meta=None):
        """Update model with new architecture-performance pair"""
        # Add workload and hardware features to arch_dict
        if workload_meta:
            arch_dict.update(workload_meta)
        
        x = self.encode_arch(arch_dict)
        self.X.append(x)
        self.y.append(performance_score)
        
        # Retrain models if we have enough data
        if len(self.X) >= 5:
            self._train_models()

    def _train_models(self):
        """Train all models in the ensemble"""
        if len(self.X) < 5:
            return
        
        X_scaled = self.scaler.fit_transform(self.X)
        
        for name, model in self.models.items():
            try:
                if name == 'gaussian_process' and len(self.X) < 10:
                    continue  # GP needs more data
                model.fit(X_scaled, self.y)
            except Exception as e:
                print(f"Warning: Failed to train {name}: {e}")
        
        self.trained = True
        
        # Update model weights based on cross-validation performance
        self._update_weights()

    def _update_weights(self):
        """Update model weights based on cross-validation performance"""
        if len(self.X) < 10:
            return
        
        X_scaled = self.scaler.transform(self.X)
        weights = {}
        
        for name, model in self.models.items():
            try:
                if name == 'gaussian_process':
                    # GP doesn't support cross_val_score easily, use default weight
                    weights[name] = 0.2
                else:
                    scores = cross_val_score(model, X_scaled, self.y, cv=min(3, len(self.X)//2))
                    weights[name] = np.mean(scores)
            except:
                weights[name] = 0.1
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.model_weights = {k: v/total_weight for k, v in weights.items()}

    def predict(self, arch_dict, workload_meta=None):
        """Predict performance with uncertainty estimation"""
        if not self.trained:
            return 0.0, 1.0  # High uncertainty if not trained
        
        if workload_meta:
            arch_dict.update(workload_meta)
        
        x = self.encode_arch(arch_dict)
        x_scaled = self.scaler.transform([x])
        
        predictions = []
        uncertainties = []
        
        for name, model in self.models.items():
            try:
                if name == 'gaussian_process':
                    pred, std = model.predict(x_scaled, return_std=True)
                    predictions.append(pred[0])
                    uncertainties.append(std[0])
                else:
                    pred = model.predict(x_scaled)[0]
                    predictions.append(pred)
                    # Estimate uncertainty using model variance (for ensemble)
                    uncertainties.append(np.std([pred]))  # Placeholder
            except:
                continue
        
        if not predictions:
            return 0.0, 1.0
        
        # Weighted ensemble prediction
        weighted_pred = sum(p * self.model_weights.get(name, 0.1) 
                           for p, name in zip(predictions, self.models.keys()))
        
        # Uncertainty estimation
        if len(uncertainties) > 1:
            # Use variance of predictions as uncertainty measure
            uncertainty = np.std(predictions)
        else:
            uncertainty = uncertainties[0] if uncertainties else 1.0
        
        return weighted_pred, uncertainty

    def suggest(self, n_candidates=20, workload_meta=None, exploration_weight=0.3):
        """Suggest architectures with active learning"""
        suggestions = []
        
        for _ in range(n_candidates):
            arch = self._generate_random_arch()
            predicted_perf, uncertainty = self.predict(arch, workload_meta)
            
            # Active learning: balance exploration vs exploitation
            exploration_score = uncertainty * exploration_weight
            exploitation_score = predicted_perf * (1 - exploration_weight)
            combined_score = exploration_score + exploitation_score
            
            suggestions.append((arch, combined_score, predicted_perf, uncertainty))
        
        # Sort by combined score
        suggestions.sort(key=lambda x: -x[1])
        
        # Store uncertainty for dashboard
        self.uncertainty_history.append([s[3] for s in suggestions[:5]])
        self.exploration_scores.append([s[1] for s in suggestions[:5]])
        
        return suggestions[:5]

    def _generate_random_arch(self):
        """Generate random architecture with expanded search space"""
        return {
            "num_layers": random.choice([1, 2, 3, 4, 5, 6]),
            "hidden_dim": random.choice([16, 32, 64, 128, 256, 512]),
            "activation": random.choice(["relu", "tanh", "leaky_relu", "elu", "swish"]),
            "regularization": random.choice(["linear", "dropout", "batch_norm"]),
            "dropout_rate": random.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
            "learning_rate": random.choice([0.0001, 0.001, 0.01, 0.1]),
            "optimizer": random.choice(["adam", "sgd", "rmsprop"]),
            "batch_size": random.choice([16, 32, 64, 128]),
            "loss_function": random.choice(["mse", "mae", "huber"]),
            "attention_heads": random.choice([0, 1, 2, 4, 8]),
            "residual_connections": random.choice([True, False]),
            "layer_normalization": random.choice([True, False]),
        }

    def get_model_performance(self):
        """Get performance metrics for each model in ensemble"""
        if not self.trained or len(self.X) < 10:
            return {}
        
        X_scaled = self.scaler.transform(self.X)
        performance = {}
        
        for name, model in self.models.items():
            try:
                if name == 'gaussian_process':
                    performance[name] = {'r2': 0.5}  # Placeholder
                else:
                    scores = cross_val_score(model, X_scaled, self.y, cv=min(3, len(self.X)//2))
                    performance[name] = {
                        'r2': np.mean(scores),
                        'std': np.std(scores)
                    }
            except:
                performance[name] = {'r2': 0.0, 'std': 0.0}
        
        return performance

# Backward compatibility
SurrogateModel = EnsembleSurrogateModel

import torch.nn as nn
import torch

class NeuralSchedulerPolicy:
    def __init__(self, model):
        self.model = model

    def assign_tasks(self, tasks, cores, current_time):
        assignments = []
        for task in tasks:
            x = self.encode_input(task, cores, current_time)
            x = x.unsqueeze(0)  # Ensure 2D input for BatchNorm1d
            core_index = self.model(x).argmax().item()
            if cores[core_index].current_task is None:
                assignments.append((task, cores[core_index]))
        return assignments

    def encode_input(self, task, cores, time):
        # Simple example: arrival, duration, num_free_cores
        free_cores = sum(1 for c in cores if c.current_task is None)
        return torch.tensor([task.arrival_time, task.duration, free_cores], dtype=torch.float32)

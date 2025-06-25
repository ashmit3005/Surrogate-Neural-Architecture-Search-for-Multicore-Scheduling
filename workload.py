import random
from simulator import Task

def generate_workload(scenario="balanced", num_tasks=20, seed=None):
    if seed is not None:
        random.seed(seed)

    tasks = []
    for i in range(num_tasks):
        if scenario == "balanced":
            arrival = i
            duration = random.randint(2, 5)
            task = Task(i, arrival, duration)

        elif scenario == "bursty":
            burst_start = (i // 5) * 10
            arrival = burst_start + random.randint(0, 2)
            duration = random.randint(1, 4)
            task = Task(i, arrival, duration)

        elif scenario == "real_time":
            arrival = i
            duration = random.randint(1, 3)
            priority = random.choice([1, 2])  # Lower = higher priority
            deadline = arrival + duration + random.randint(1, 3)
            task = Task(i, arrival, duration, priority)
            task.deadline = deadline

        elif scenario == "memory_bound":
            arrival = i
            duration = random.randint(2, 4)
            memory_demand = random.randint(1, 3)  # GB or relative scale
            task = Task(i, arrival, duration)
            task.memory_demand = memory_demand

        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        tasks.append(task)

    return sorted(tasks, key=lambda t: t.arrival_time)

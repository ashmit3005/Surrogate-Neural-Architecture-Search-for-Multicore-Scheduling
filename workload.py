import random
import numpy as np
from simulator import Task

def generate_workload(scenario="balanced", num_tasks=20, seed=None, complexity_level="basic", **advanced_options):
    if seed is not None:
        random.seed(seed)

    tasks = []
    
    # Extract advanced options
    stochastic_arrivals = advanced_options.get('stochastic_arrivals', False)
    arrival_jitter = advanced_options.get('arrival_jitter', 0.5)
    preemption_enabled = advanced_options.get('preemption_enabled', False)
    preemption_frequency = advanced_options.get('preemption_frequency', 0.1)
    
    if complexity_level == "basic":
        # Original workload generation
        for i in range(num_tasks):
            if scenario == "balanced":
                arrival = i
                if stochastic_arrivals:
                    arrival += random.uniform(-arrival_jitter, arrival_jitter)
                duration = random.randint(2, 5)
                task = Task(i, arrival, duration)

            elif scenario == "bursty":
                burst_start = (i // 5) * 10
                arrival = burst_start + random.randint(0, 2)
                if stochastic_arrivals:
                    arrival += random.uniform(-arrival_jitter, arrival_jitter)
                duration = random.randint(1, 4)
                task = Task(i, arrival, duration)

            elif scenario == "real_time":
                arrival = i
                if stochastic_arrivals:
                    arrival += random.uniform(-arrival_jitter, arrival_jitter)
                duration = random.randint(1, 3)
                priority = random.choice([1, 2])
                deadline = arrival + duration + random.randint(1, 3)
                task = Task(i, arrival, duration, priority)
                task.deadline = deadline

            elif scenario == "memory_bound":
                arrival = i
                if stochastic_arrivals:
                    arrival += random.uniform(-arrival_jitter, arrival_jitter)
                duration = random.randint(2, 4)
                memory_demand = random.randint(1, 3)
                task = Task(i, arrival, duration)
                task.memory_demand = memory_demand

            # Set preemption based on advanced options
            if preemption_enabled:
                task.preemptible = random.random() < preemption_frequency

            tasks.append(task)
    
    elif complexity_level == "advanced":
        # Advanced workload with dependencies, heterogeneous tasks, and realistic patterns
        for i in range(num_tasks):
            # Realistic arrival patterns
            if scenario == "balanced":
                arrival = i + random.randint(-1, 1)  # Some jitter
                if stochastic_arrivals:
                    arrival += random.uniform(-arrival_jitter, arrival_jitter)
            elif scenario == "bursty":
                burst_start = (i // 5) * 10
                arrival = burst_start + random.randint(0, 3)
                if stochastic_arrivals:
                    arrival += random.uniform(-arrival_jitter, arrival_jitter)
            elif scenario == "real_time":
                arrival = i + random.randint(-2, 2)
                if stochastic_arrivals:
                    arrival += random.uniform(-arrival_jitter, arrival_jitter)
            elif scenario == "memory_bound":
                arrival = i + random.randint(-1, 1)
                if stochastic_arrivals:
                    arrival += random.uniform(-arrival_jitter, arrival_jitter)
            
            # Task characteristics based on scenario
            if scenario == "real_time":
                duration = random.randint(1, 4)
                priority = random.choice([1, 2, 3])  # More priority levels
                deadline = arrival + duration + random.randint(1, 5)
                task = Task(i, arrival, duration, priority)
                task.deadline = deadline
                task.preemptible = random.choice([True, False]) if preemption_enabled else False
                
            elif scenario == "memory_bound":
                duration = random.randint(2, 6)
                memory_demand = random.randint(2, 8)
                task = Task(i, arrival, duration)
                task.memory_demand = memory_demand
                task.cpu_intensity = random.uniform(0.3, 0.7)  # Lower CPU intensity
                
            else:  # balanced or bursty
                duration = random.randint(2, 6)
                task = Task(i, arrival, duration)
                task.cpu_intensity = random.uniform(0.7, 1.3)
            
            # Add dependencies (DAG structure)
            if i > 0 and random.random() < 0.3:  # 30% chance of dependency
                num_deps = random.randint(1, min(3, i))
                deps = random.sample(range(i), num_deps)
                task.dependencies = deps
            
            # Add affinity preferences
            if random.random() < 0.2:  # 20% chance of core affinity
                task.affinity_cores = random.sample(range(4), random.randint(1, 2))
            
            # Set preemption based on advanced options
            if preemption_enabled:
                task.preemptible = random.random() < preemption_frequency
            
            tasks.append(task)
    
    elif complexity_level == "enterprise":
        # Enterprise-level workload with complex patterns
        # Create task clusters with dependencies
        num_clusters = max(1, num_tasks // 10)
        cluster_size = num_tasks // num_clusters
        
        for cluster_id in range(num_clusters):
            cluster_start = cluster_id * cluster_size
            cluster_end = min((cluster_id + 1) * cluster_size, num_tasks)
            
            # Create cluster tasks
            for i in range(cluster_start, cluster_end):
                # Arrival pattern: tasks in cluster arrive close together
                base_arrival = cluster_id * 15
                arrival = base_arrival + random.randint(0, 5)
                
                # Task types within cluster
                task_type = random.choice(["compute", "io", "memory", "mixed"])
                
                if task_type == "compute":
                    duration = random.randint(3, 8)
                    task = Task(i, arrival, duration)
                    task.cpu_intensity = random.uniform(1.2, 2.0)
                    task.cache_footprint = random.randint(5, 15)
                    task.io_operations = random.randint(0, 2)
                    
                elif task_type == "io":
                    duration = random.randint(2, 5)
                    task = Task(i, arrival, duration)
                    task.cpu_intensity = random.uniform(0.3, 0.6)
                    task.io_operations = random.randint(3, 8)
                    task.memory_demand = random.randint(1, 3)
                    
                elif task_type == "memory":
                    duration = random.randint(4, 10)
                    task = Task(i, arrival, duration)
                    task.memory_demand = random.randint(4, 12)
                    task.cache_footprint = random.randint(8, 20)
                    task.cpu_intensity = random.uniform(0.8, 1.2)
                    
                else:  # mixed
                    duration = random.randint(3, 7)
                    task = Task(i, arrival, duration)
                    task.cpu_intensity = random.uniform(0.7, 1.4)
                    task.memory_demand = random.randint(2, 6)
                    task.io_operations = random.randint(1, 4)
                
                # Add dependencies within cluster
                if i > cluster_start and random.random() < 0.4:
                    deps = random.sample(range(cluster_start, i), min(2, i - cluster_start))
                    task.dependencies = deps
                
                # Add cross-cluster dependencies
                if cluster_id > 0 and random.random() < 0.1:
                    prev_cluster_start = (cluster_id - 1) * cluster_size
                    prev_cluster_end = cluster_id * cluster_size
                    cross_dep = random.randint(prev_cluster_start, prev_cluster_end - 1)
                    task.dependencies = task.dependencies + [cross_dep]
                
                # Priority and deadlines for real-time scenarios
                if scenario == "real_time":
                    task.priority = random.choice([1, 2, 3, 4])
                    task.deadline = arrival + duration + random.randint(2, 8)
                    task.preemptible = random.choice([True, False])
                
                tasks.append(task)
    
    return sorted(tasks, key=lambda t: t.arrival_time)

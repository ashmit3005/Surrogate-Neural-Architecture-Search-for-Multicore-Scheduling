import heapq
from collections import deque, defaultdict
import random
import math
from typing import List, Dict, Optional, Tuple
import numpy as np

class Task:
    def __init__(self, task_id, arrival_time, duration, priority=1):
        self.task_id = task_id
        self.arrival_time = arrival_time
        self.duration = duration
        self.remaining_time = duration
        self.priority = priority
        self.start_time = None
        self.end_time = None
        self.deadline = None           # Optional
        self.memory_demand = None      # Optional
        
        # New advanced features
        self.dependencies = []  # List of task_ids that must complete first
        self.cpu_intensity = random.uniform(0.5, 1.5)  # CPU vs I/O intensity
        self.cache_footprint = random.randint(1, 10)  # Cache usage in MB
        self.io_operations = random.randint(0, 5)  # Number of I/O operations
        self.energy_per_unit = random.uniform(0.1, 2.0)  # Energy consumption per time unit
        self.preemptible = random.choice([True, False])  # Can be preempted
        self.migration_cost = random.uniform(0.1, 1.0)  # Cost to migrate task
        self.affinity_cores = []  # Preferred cores for this task

class Core:
    def __init__(self, core_id, core_type="cpu", memory_capacity=4, speed=1.0):
        self.core_id = core_id
        self.core_type = core_type  # "cpu", "gpu", "efficient", "performance"
        self.memory_capacity = memory_capacity
        self.speed = speed
        self.current_task = None
        self.time_remaining = 0
        self.busy_time = 0
        
        # New advanced features
        self.energy_consumption = 0.0
        self.cache_size = 8 if core_type == "performance" else 4  # MB
        self.memory_bandwidth = 25.6 if core_type == "performance" else 12.8  # GB/s
        self.io_bandwidth = 3.2 if core_type == "performance" else 1.6  # GB/s
        self.thermal_state = 0.0  # Temperature factor (0-1)
        self.preemption_count = 0
        self.migration_count = 0
        
        # Resource contention tracking
        self.cache_misses = 0
        self.memory_wait_time = 0
        self.io_wait_time = 0

    def assign(self, task: Task, current_time):
        if self.current_task and self.current_task.preemptible:
            # Preempt current task
            self.preemption_count += 1
            self.current_task.remaining_time = self.time_remaining
            self.current_task.end_time = None
            self.current_task.start_time = None
        
        self.current_task = task
        self.time_remaining = task.remaining_time / self.speed
        task.start_time = current_time
        task.core_id = self.core_id
        
        # Energy consumption
        self.energy_consumption += task.energy_per_unit * self.speed

    def step(self, current_time):
        if self.current_task:
            # Simulate resource contention
            cache_penalty = 0
            if self.current_task.cache_footprint > self.cache_size:
                cache_penalty = 0.2
                self.cache_misses += 1
            
            memory_penalty = 0
            if self.current_task.memory_demand and self.current_task.memory_demand > self.memory_capacity * 0.8:
                memory_penalty = 0.3
                self.memory_wait_time += 1
            
            io_penalty = 0
            if self.current_task.io_operations > 0:
                io_penalty = 0.1 * self.current_task.io_operations
                self.io_wait_time += 1
            
            # Thermal effects
            thermal_penalty = self.thermal_state * 0.1
            
            # Calculate effective work unit
            effective_speed = self.speed * (1 - cache_penalty - memory_penalty - io_penalty - thermal_penalty)
            effective_speed = max(effective_speed, 0.1)  # Minimum speed
            
            self.time_remaining -= effective_speed
            self.busy_time += 1
            self.current_task.remaining_time -= effective_speed
            
            # Update thermal state
            self.thermal_state = min(1.0, self.thermal_state + 0.01)
            
            if self.time_remaining <= 0:
                self.current_task.end_time = current_time + 1
                finished_task = self.current_task
                self.current_task = None
                return finished_task
        else:
            # Cool down when idle
            self.thermal_state = max(0.0, self.thermal_state - 0.02)
        
        return None

class AdvancedSimulator:
    def __init__(self, num_cores, scheduling_policy, verbose=False):
        # Create heterogeneous cores
        self.cores = []
        for i in range(num_cores):
            if i < num_cores // 4:
                core_type = "performance"
                speed = 2.0
                memory = 8
            elif i < num_cores // 2:
                core_type = "efficient"
                speed = 0.7
                memory = 4
            else:
                core_type = "cpu"
                speed = 1.0
                memory = 6
            
            self.cores.append(Core(i, core_type, memory, speed))
        
        self.task_queue = deque()
        self.completed_tasks = []
        self.time = 0
        self.policy = scheduling_policy
        self.verbose = verbose
        
        # Advanced tracking
        self.total_energy = 0.0
        self.total_preemptions = 0
        self.total_migrations = 0
        self.resource_contention_stats = {
            'cache_misses': 0,
            'memory_wait_time': 0,
            'io_wait_time': 0
        }
        
        # DAG dependency tracking
        self.task_dependencies = {}
        self.ready_tasks = set()
        self.running_tasks = set()

    def load_tasks(self, tasks):
        # Sort by arrival time and build dependency graph
        sorted_tasks = sorted(tasks, key=lambda t: t.arrival_time)
        self.task_queue = deque(sorted_tasks)
        
        # Build dependency graph
        for task in tasks:
            self.task_dependencies[task.task_id] = set(task.dependencies)
            if not task.dependencies:
                self.ready_tasks.add(task.task_id)

    def check_dependencies(self, task):
        """Check if all dependencies are satisfied"""
        if not task.dependencies:
            return True
        return all(dep_id in [t.task_id for t in self.completed_tasks] for dep_id in task.dependencies)

    def run(self, max_time=1000):
        active_tasks = []
        
        while self.time < max_time:
            # Add tasks that arrive at this time
            while self.task_queue and self.task_queue[0].arrival_time <= self.time:
                task = self.task_queue.popleft()
                if self.check_dependencies(task):
                    active_tasks.append(task)
                    self.ready_tasks.add(task.task_id)

            # Ask the policy to decide task-core mapping
            assignments = self.policy.assign_tasks(active_tasks, self.cores, self.time)

            for task, core in assignments:
                if task.memory_demand and task.memory_demand > core.memory_capacity:
                    continue  # skip if core can't handle the task
                
                # Check if task is already running (migration)
                if task.task_id in self.running_tasks:
                    self.total_migrations += 1
                
                core.assign(task, self.time)
                active_tasks.remove(task)
                self.running_tasks.add(task.task_id)

            # Step all cores
            for core in self.cores:
                finished = core.step(self.time)
                if finished:
                    self.completed_tasks.append(finished)
                    self.running_tasks.discard(finished.task_id)
                    self.total_energy += core.energy_consumption
                    self.total_preemptions += core.preemption_count
                    
                    # Update resource contention stats
                    self.resource_contention_stats['cache_misses'] += core.cache_misses
                    self.resource_contention_stats['memory_wait_time'] += core.memory_wait_time
                    self.resource_contention_stats['io_wait_time'] += core.io_wait_time

            self.time += 1

        return self.evaluate()

    def evaluate(self):
        latencies = [(t.end_time - t.arrival_time) for t in self.completed_tasks if t.end_time]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        deadline_misses = sum(
            1 for t in self.completed_tasks if t.deadline and t.end_time and t.end_time > t.deadline
        )

        utilizations = [c.busy_time / self.time for c in self.cores]
        avg_util = sum(utilizations) / len(utilizations)

        # Energy efficiency
        energy_efficiency = self.total_energy / len(self.completed_tasks) if self.completed_tasks else 0

        # Resource contention metrics
        avg_cache_misses = self.resource_contention_stats['cache_misses'] / len(self.cores)
        avg_memory_wait = self.resource_contention_stats['memory_wait_time'] / len(self.cores)
        avg_io_wait = self.resource_contention_stats['io_wait_time'] / len(self.cores)

        return {
            "avg_latency": avg_latency,
            "avg_utilization": avg_util,
            "num_completed": len(self.completed_tasks),
            "deadline_misses": deadline_misses,
            "total_energy": self.total_energy,
            "energy_efficiency": energy_efficiency,
            "total_preemptions": self.total_preemptions,
            "total_migrations": self.total_migrations,
            "avg_cache_misses": avg_cache_misses,
            "avg_memory_wait": avg_memory_wait,
            "avg_io_wait": avg_io_wait,
            "resource_contention_score": (avg_cache_misses + avg_memory_wait + avg_io_wait) / 3
        }

# Backward compatibility
Simulator = AdvancedSimulator
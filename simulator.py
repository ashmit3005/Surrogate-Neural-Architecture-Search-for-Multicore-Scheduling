import heapq
from collections import deque, defaultdict

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

class Core:
    def __init__(self, core_id, memory_capacity=4, speed=1.0):
        self.core_id = core_id
        self.memory_capacity = memory_capacity  # new
        self.speed = speed                      # new (used later)
        self.current_task = None
        self.time_remaining = 0
        self.busy_time = 0

    def assign(self, task: Task, current_time):
        self.current_task = task
        self.time_remaining = task.remaining_time
        task.start_time = current_time
        task.core_id = self.core_id

    def step(self, current_time):
        if self.current_task:
            work_unit = self.speed  # could be < 1.0 for slower cores
            self.time_remaining -= work_unit
            self.busy_time += 1  # still count clock cycles
            self.current_task.remaining_time -= work_unit
            if self.time_remaining <= 0:
                self.current_task.end_time = current_time + 1
                finished_task = self.current_task
                self.current_task = None
                return finished_task
        return None


class Simulator:
    def __init__(self, num_cores, scheduling_policy, verbose=False):
        self.cores = [Core(i) for i in range(num_cores)]
        self.task_queue = deque()
        self.completed_tasks = []
        self.time = 0
        self.policy = scheduling_policy
        self.verbose = verbose

    def load_tasks(self, tasks):
        self.task_queue = deque(sorted(tasks, key=lambda t: t.arrival_time))

    def run(self, max_time=1000):
        active_tasks = []
        while self.time < max_time:
            # Add tasks that arrive at this time
            while self.task_queue and self.task_queue[0].arrival_time <= self.time:
                active_tasks.append(self.task_queue.popleft())

            # Ask the policy to decide task-core mapping
            assignments = self.policy.assign_tasks(active_tasks, self.cores, self.time)


            for task, core in assignments:
                if task.memory_demand and task.memory_demand > core.memory_capacity:
                    continue  # skip if core can't handle the task
                core.assign(task, self.time)
                active_tasks.remove(task)

            # Step all cores
            for core in self.cores:
                finished = core.step(self.time)
                if finished:
                    self.completed_tasks.append(finished)

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

        return {
            "avg_latency": avg_latency,
            "avg_utilization": avg_util,
            "num_completed": len(self.completed_tasks),
            "deadline_misses": deadline_misses,
        }
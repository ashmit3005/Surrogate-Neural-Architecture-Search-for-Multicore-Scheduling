class RoundRobinPolicy:
    def __init__(self):
        self.next_core = 0

    def assign_tasks(self, tasks, cores, current_time):
        assignments = []
        for task in tasks:
            # Find a free core
            for _ in range(len(cores)):
                core = cores[self.next_core]
                self.next_core = (self.next_core + 1) % len(cores)
                if core.current_task is None:
                    assignments.append((task, core))
                    break
        return assignments

import ray


@ray.remote
class Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        print(f"Before Incremented: {self.value}")  # shows in console and ray dashboard
        self.value += 1
        print(f"After Incremented: {self.value}")
        return self.value


# Create an actor
counter = Counter.remote()
obj_ref = counter.increment.remote()
print(ray.get(obj_ref))

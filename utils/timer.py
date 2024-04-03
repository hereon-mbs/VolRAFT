import time

## TODO: update this file
## Use this for now

class Timer(object):
    def __init__(self, name=None):
        self.start_time = None
        self.end_time = None
        self.name = name

    def __enter__(self):
        self.start(verbose=False)

    def __exit__(self, type, value, traceback):
        if self.name:
            print(f'[{self.name}]', end=" ")
        print(f'Elapsed time (seconds): {time.time() - self.start_time:.6f}')

    def start(self, verbose=True):
        self.start_time = time.time()
        if verbose:
            print("Timer started")

    def stop(self, verbose=True):
        self.end_time = time.time()
        if verbose:
            print("Timer stopped")

    def elapsed_time(self, verbose=True):
        if self.start_time is None:
            print("Timer has not been started")
            return None
        elif self.end_time is None:
            print("Timer has not been stopped")
            return None
        else:
            elapsed = self.end_time - self.start_time
            if verbose:
                if self.name:
                    print(f'[{self.name}]', end=" ")
                print(f"Elapsed time (seconds): {elapsed:.6f}")
            return elapsed

if __name__ == "__main__":
    # Example usage:
    # Create an instance of the Timer class
    # Start the timer, perform some operations, and then stop the timer to calculate elapsed time
    print("--- Example usage ---")
    timer = Timer('example 1')
    timer.start()
    
    for n in range(5000):
        # Perform some operations...
        count = n

    timer.stop()
    elapsed = timer.elapsed_time()

    print("--- Example usage ---")
    # Perform some operations...
    with Timer('example 2'):
        for n in range(5000):
            # Perform some operations...
            count = n

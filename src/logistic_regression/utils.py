import time

class Timer:
    def tic(self, msg):
        self.tic = time.perf_counter()
        print(msg)

    def toc(self, msg):
        self.toc = time.perf_counter()
        print(msg)
        print(f"Total time: {self.toc - self.tic:0.4f} seconds")

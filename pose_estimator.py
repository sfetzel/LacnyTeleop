from abc import ABC, abstractmethod
import threading
import time
import numpy as np

class PoseEstimator(ABC):
    def __init__(self):
        self.latest_deltas = None
        self.thread = None
        self.stop_requested = False
        
    @abstractmethod
    def run(self):
        pass
    
    def get_deltas(self):
        result = self.latest_deltas
        self.latest_deltas = None
        return result
        
    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self.run)
            self.thread.start()
            self.stop_requested = False
            
    def stop(self):
        if not self.thread is None:
            self.stop_requested = True
            self.thread.join()

class MockEstimator(PoseEstimator):
    def run(self):
        while not self.stop_requested:
            time.sleep(0.1)
            self.latest_deltas = np.concatenate([-np.array([0.1, 0.05, 0.01]), np.array([0.1, 0, 0])])


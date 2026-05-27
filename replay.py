from collections import deque
import pickle
import random

class ReplayBuffer:
    def __init__(self,maxlen=50000):
        self.buffer = deque(maxlen=maxlen)

    def push(self,board_tensor,policy,value):
        self.buffer.append((board_tensor,policy,value))

    def sample(self,batch_size):
        return random.sample(self.buffer,batch_size)

    def __len__(self):
        return len(self.buffer)

    def save(self,path="replay.pkl"):
        with open(path,"wb") as f:
            pickle.dump(self.buffer,f)

    def load(self,path="replay.pkl"):
        with open(path,"rb") as f:
            self.buffer = pickle.load(f)
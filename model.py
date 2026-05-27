import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels,channels,3,padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels,channels,3,padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self,x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)
    
class ChessNet(nn.Module):
    def __init__(self,channels=64,res_blocks=5):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(12,channels,3,padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(*[ResBlock(channels) for _ in range(res_blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels,2,1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2*8*8,4096)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels,1,1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1*8*8,64),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.Tanh()
        )

    def forward(self,x):
        x = self.entry(x)
        x = self.res_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

def save(model,path="chessnet.pth"):
    torch.save(model.state_dict(),path)

def load(model,path="chessnet.pth"):
    model = ChessNet()
    model.load_state_dict(torch.load(path))
    return model
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ChessNet(nn.Module):
    def __init__(self, channels=64, res_blocks=5):
        super().__init__()
        # 13 input planes: 6 white pieces + 6 black pieces + 1 turn plane
        self.entry = nn.Sequential(
            nn.Conv2d(13, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(res_blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 4096),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.entry(x)
        x = self.body(x)
        return self.policy_head(x), self.value_head(x)

    def predict(self, board_tensor):
        """Single-board inference. Returns (policy_probs numpy array, value float)."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            x = torch.tensor(board_tensor).unsqueeze(0)
            policy_logits, value = self(x)
            policy = torch.softmax(policy_logits, dim=1).squeeze().numpy()
        if was_training:
            self.train()
        return policy, value.item()


def save(model, path="chessnet.pth"):
    torch.save(model.state_dict(), path)


def load(path="chessnet.pth", channels=64, res_blocks=5):
    model = ChessNet(channels=channels, res_blocks=res_blocks)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model

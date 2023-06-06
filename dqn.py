import torch 

class DQN(torch.nn.Module):
    def __init__(self, n_frames, conv1, conv2, conv3, k_size1, k_size2, k_size3, stride1, stride2, stride3, fc1_size, fc2_size):
        super(DQN, self).__init__()
        self.c1 = torch.nn.Conv2d(in_channels=n_frames, out_channels = conv1, kernel_size=k_size1,stride=stride1)
        self.c2 = torch.nn.Conv2d(in_channels=conv1, out_channels = conv2 ,kernel_size=k_size2,stride=stride2)
        self.c3 = torch.nn.Conv2d(in_channels=conv2, out_channels = conv3 ,kernel_size=k_size3,stride=stride3)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(3136, fc1_size)
        self.fc2 = torch.nn.Linear(fc1_size, fc2_size)
        self.relu = torch.nn.ReLU()
        
    def forward(self, input):
        out_conv = self.c1(input)
        out_conv = self.relu(out_conv)
        out_conv = self.c2(out_conv)
        out_conv = self.relu(out_conv)
        out_conv = self.c3(out_conv)
        out_conv = self.relu(out_conv)
        flat = self.flatten(out_conv)
        out = self.fc1(flat)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out

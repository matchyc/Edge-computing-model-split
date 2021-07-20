import torch
import torch.nn as nn
import torch.tensor as tensor
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
torch.__version__



class Manager():
    def __init__(self) -> None:
        self.BATCH_SIZE=512 
        self.EPOCHS=2
        self.input_data = None
        self.label = None
    def prepare_data(self, model):
        # self.train_loader = torch.utils.data.DataLoader(
        #                     datasets.MNIST('data', train=True, download=True, 
        #                     transform=transforms.Compose([
        #                     transforms.ToTensor(),
        #                     transforms.Normalize((0.1307,), (0.3081,))
        #                     ])),
        #                     batch_size=self.BATCH_SIZE, shuffle=True)
        if model == 'mnist':
            self.test_loader = torch.utils.data.DataLoader(
                            datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                            ])),
                            batch_size=self.BATCH_SIZE, shuffle=True)

        
    def load_params(self, model, path):
        model.load_state_dict(torch.load(path))
        


class ConvNet_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_list = nn.ModuleList([
            nn.Conv2d(1, 10, 5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(10, 20, 3),
            nn.ReLU(),
            nn.Linear(20*10*10, 500),
            nn.ReLU(),
            nn.Linear(500, 10),
            nn.LogSoftmax(dim=1)
        ])

        self.layer_size = len(self.layer_list)

    def forward(self, x, start, end):
        in_size = x.size(0)
        out = x
        for i in range(start, end):
            print(i)
            out = self.layer_list[i](out)
            if i == 4:
                out = out.view(in_size, -1)
        return out


manager = Manager()
model_minist = ConvNet_MNIST()

model_dict = {'mnist': model_minist} # could be added more model



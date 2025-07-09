import logging
import torch
import logging
import torch.nn as nn

from tvm_common import get_common_parser, convert

class TorchModel(nn.Module):
    def __init__(self):
        super(TorchModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x
    
def load_torch_model() -> torch.nn.Module:
    model = TorchModel()
    model.eval()
    model.to(torch.float32)
    return model

def main(args):
    model = load_torch_model()
    example_args = (torch.randn(1, 784, dtype=torch.float32),)
    logging.info("Convert pytorch model to tvm")
    convert(args, model, example_args)

if __name__ == "__main__":
    parser = get_common_parser()
    args = parser.parse_args()
    main(args)
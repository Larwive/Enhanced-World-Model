import argparse
import torch

#  Only support single device for now.
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

parser = argparse.ArgumentParser()

# Data

# Training Process ########
parser.add_argument('--random-seed', type=int, default=42)
parser.add_argument('--max-epoch', type=int, default=200)

# Number of consecutive epochs without increase in accuracy of validation set before early stopping
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--learning-rate', type=float, default=1e-3)
parser.add_argument('--dropout', type=float, default=0.2)

parser.add_argument('--save-path', default='../saved_models/')
parser.add_argument('--load-path', default='TOFILL')
parser.add_argument('--gpu', default='0')

args = parser.parse_args()

print(args)
print(device)

import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help='path to .pth.tar file')
parser.add_argument('output', type=str, help='output path')

args = parser.parse_args()

checkpoint = torch.load(args.path)
torch.save(checkpoint['state_dict'], args.output)

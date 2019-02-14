import argparse
import collections
import torch

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help='path to .pth.tar file')
parser.add_argument('output', type=str, help='output path')

args = parser.parse_args()

checkpoint = torch.load(args.path)
old_dict = checkpoint['state_dict']
new_dict = collections.OrderedDict()


for k in old_dict.keys():
    new_k = k.replace("module.", "")
    new_dict[new_k] = old_dict[k]

torch.save(new_dict, args.output)

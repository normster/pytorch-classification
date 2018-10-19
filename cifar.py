'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
import argparse
import copy
import os
import pickle
import shutil
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=100, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=1000, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--batch-schedule', default='batch_original', type=str)
parser.add_argument('--width', default=10, type=int)
parser.add_argument('--steps', default=4, type=int)
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--visualize-loss', action='store_true',
                    help='visualize loss landscape')
parser.add_argument('--vector', default='', type=str, metavar='PATH',
                    help='path to serialized perturbation vector')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)

random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.manual_seed_all(args.manualSeed)
else:
    device = torch.device('cpu')

best_acc = 0  # best test accuracy

def batch_original(epoch, width, steps):
    return args.train_batch

def batch_cbs(epoch, width, steps):
    k = (epoch // width) % steps
    return int(2 ** k) * args.train_batch

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)



    # Data
    print('==> Preparing dataset {}'.format(args.dataset))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100


    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    model = model.to(device)
    cudnn.benchmark = True
    print('    Total params: {0:.2f}M'.format(sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        c_fname = os.path.join(args.checkpoint, 'log.txt')
        if os.path.isfile(c_fname):
            if input("Overwrite existing checkpoint ({})? (y/n): ".format(c_fname)) != 'y':
                return
        logger = Logger(c_fname, title=title)
        logger.set_names(['Learning Rate', "Batch Size", 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch)
        print(' Test Loss:  {0:.8f}, Test Acc:  {1:.2f}'.format(test_loss, test_acc))
        return

    if args.visualize_loss:
        print('==> Visualizing loss landscape..')
        vector = None
        if args.vector:
            with open(args.vector, 'r') as f:
                vector = pickle.load(f)    
        visualize_loss(model, testloader, trainloader, -0.1, 0.1, vector)
        return

    compute_batch = globals()[args.batch_schedule]
    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        batch_size = compute_batch(epoch, args.width, args.steps)
        batch_sampler = data.BatchSampler(trainloader.sampler, batch_size, False)
        trainloader.batch_sampler = batch_sampler

        print('\nEpoch: [{} | {}] LR: {}'.format(epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch)
        test_loss, test_acc = test(testloader, model, criterion, epoch)

        # append logger file
        logger.append([state['lr'], batch_size, train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)

def visualize_loss(model, testloader, trainloader, left, right, vector, samples=200, output_dir=""):
    if not vector:
        vector = []
        for p in model.parameters():
            vector.append(np.random.normal(size=p.shape))
        print('\tSaving perturbation_vector')
        with open('perturbation_vector', 'w') as f:
            pickle.dump(vector, f)

    test_losses = []
    test_acces = []

    train_losses = []
    train_acces = []

    epses = np.linspace(left, right, num=samples)
    for i, eps in enumerate(epses):
        print("\tTesting perturbation {}/{}".format(i+1, samples))
        pert_model = perturb(model, vector, eps)
        test_loss, test_acc = test(testloader, pert_model, nn.CrossEntropyLoss(), 0)
        train_loss, train_acc = test(trainloader, pert_model, nn.CrossEntropyLoss(), 0)

        test_losses.append(test_loss)
        test_acces.append(test_acc)

        train_losses.append(train_loss)
        train_acces.append(train_acc)

    plt.plot(epses, train_losses, label="Train Loss")
    plt.plot(epses, test_losses, label="Test Loss")
    plt.xlabel(u"\u03B5 (\u03B8' = \u03B8 + \u03B5)")
    plt.ylabel("Cross Entropy Loss")
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "loss.pdf"), dpi=150)

    plt.clf()

    plt.plot(epses, train_acces, label="Train Acc")
    plt.plot(epses, test_acces, label="Test Acc")
    plt.xlabel(u"\u03B5 (\u03B8' = \u03B8 + \u03B5)")
    plt.ylabel("Top 1 Accuracy %")
    plt.axis([left, right, 0, 100]) 
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "acc.pdf"), dpi=150)


def perturb(model, vector, eps):
    ret = copy.deepcopy(model)

    for param, v_i in zip(ret.parameters(), vector):
        param.requires_grad = False
        update = torch.from_numpy(eps * v_i).to(device)
        param += update.float() 

    return ret

def train(trainloader, model, criterion, optimizer, epoch):
    # switch to train mode
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs = inputs.to(device)
        targets = targets.to(device)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        ))
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch):
    global best_acc

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

    print('Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        ))

    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()

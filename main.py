#!/bin/env python 
import os
import sys
import torch
import models
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import lib
from lib.util import progress_bar
from torch.autograd import Variable

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name(cifar10, cifar100).')
    parser.add_argument('--model', type=str, default='vgg', help='Model type to use.')
    parser.add_argument('--outdir', type=str, default='./log', help='Output path.')
    parser.add_argument('--aepoch', type=int, default=10, help='The number of epochs for arch learning.')
    parser.add_argument('--wepoch', type=int, default=200, help='The number of epochs for weight learning.')
    parser.add_argument('--alr', type=float, default=0.1, help='Learning rate of the architecture learning.')
    parser.add_argument('--batchsize', type=int, default=256, help='Batchsize of dataloader.')
    parser.add_argument('--expansion', type=float, default=1.0, help='The expansion ratio for the model.')
    parser.add_argument('--ratio', type=float, default=0.5, help='The prune ratio used in sparsity regularzation.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for weight training.')
    parser.add_argument('--lr_decay', action='store_true', default=False, help='If use the learning rate decay.')
    parser.add_argument('--balance', type=float, default=0.5, help='The balance constant of the sparsity regularization.')
    return parser.parse_args()


def prepare_data(args):
    cifar_train_trans = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    cifar_val_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    if args.dataset == 'cifar10':
        train_data = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=cifar_train_trans)
        val_data = datasets.CIFAR10('./data/cifar10', train=False, download=False, transform=cifar_val_trans)
    elif args.dataset == 'cifar100':
        train_data = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=cifar_train_trans)
        val_data = datasets.CIFAR100('./data/cifar100', train=False, download=False, transform=cifar_val_trans)
    else:
        raise NotImplementedError
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batchsize, shuffle=False, num_workers=8)
    return train_loader, val_loader


def regularzation_update(model, args):
    if not args.sum_channel:
        args.sum_channel = 0
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                args.sum_channel += layer.weight.size()[0]
    sumc = args.sum_channel
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.weight.grad.data.add(args.balance * 2.0 * torch.sign(layer.weight.data)*(layer.weight.data/sumc-args.ratio))


def arch_train(model, args, train_loader, val_loader):
    '''First Train the architecture parameters without updating the other weights'''
    # Freeze the weights
    for para in model.parameters():
        para.requires_grad = False
    # Enable the parameters of network architecture
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            for para in layer.parameters():
                para.requires_grad = True
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.alr)
    print('Training the Architecture')

    for epochid in range(args.aepoch):
        print('==> Epoch: %d' % epochid)
        train_loss = 0.0
        total = 0
        correct = 0
        for batchid, (data, target) in enumerate(train_loader):
            if args.Use_Cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            regularzation_update(model, args)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            avg_loss = train_loss / (batchid+1)
            acc = correct / total
            progress_bar(batchid, len(train_loader), 'Loss: %.3f | Acc: %.3f'% (avg_loss, acc))
        
            
def binary_search(model, gates, args):
    # TODO: use binary search to find the threshold for the pruning
    pos = int(len(gates) * args.ratio)
    sorted_gates, index = torch.sort(gates)
    return sorted_gates[pos]


def prune(model, args):
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    print('Pruning the network according to the architecture parameters.')
    gates = torch.zeros(args.sum_channel)
    index = 0
    pruned = 0
    cfg = []
    cfg_mask = []
    for lid, layer in enumerate(model.modules()):
        if isinstance(layer, nn.BatchNorm2d):
            nchannel = layer.weight.data.shape[0]
            gates[index:index+nchannel] = layer.weight.data.abs().clone()
            index += nchannel
    threshold = binary_search(model, gates, args)
    for lid, layer in enumerate(model.modules()):
        if isinstance(layer, nn.BatchNorm2d):
            weight_copy = layer.weight.data.abs().clone()
            mask = weight_copy.gt(threshold)
            mask = mask.float().cuda()
            layer.weight.data.mul_(mask)
            layer.bias.data.mul_(mask)
            pruned += mask.shape[0] - sum(mask)
            cfg.append(int(torch.sum(mask).item()))
            cfg_mask.append(mask)
        elif isinstance(layer, nn.MaxPool2d):
            cfg.append('M')
    print('Original channel number: ',args.sum_channel)
    print(cfg)
    print('After pruned channel number: ', sum(filter(lambda x: x!='M', cfg)))
    new_model = models.__dict__[args.model](args.num_class, cfg=cfg)
    logfile = os.path.join(args.outdir, 'log.txt')
    with open(logfile, 'w') as logf:
        logf.write('Configuration of the pruned model\n')
        logf.write(str(cfg))
    return new_model


def validation(model, val_loader, criterion, Use_Cuda):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batchid, (data, target) in enumerate(val_loader):
            if Use_Cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            avg_acc = correct / total
            avg_loss = test_loss / (batchid + 1)
            progress_bar(batchid, len(val_loader), 'Loss: %.3f | Acc: %.3f' % (avg_loss, avg_acc))
    return correct/total


def weight_train(model, train_loader, val_loader, args):
    best_acc = 0.0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.98)
    for i in range(args.wepoch):
        print('==>Epoch %d' % (i+1))
        print('==>Training')
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for batchid, (data, target) in enumerate(train_loader):
            if args.Use_Cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += output.size(0)
            correct += predicted.eq(target).sum().item()
            avg_loss = train_loss / (batchid + 1)
            avg_acc = correct / total
            progress_bar(batchid, len(train_loader), 'Loss: %.3f | Acc: %.3f' % (avg_loss, avg_acc))
        # Validation
        print('==>Validating')
        val_acc = validation(model, val_loader, criterion, args.Use_Cuda)    
        if val_acc > best_acc:
            best_acc = val_acc
            best_checkpoint = {'state_dict':model.state_dict(), 'Acc':best_acc}
            fname = os.path.join(args.outdir, 'best.pth.tar')
            torch.save(best_checkpoint, fname)
        print('==>Best validation accuracy', best_acc)
        # Save checkpoint
        if (i + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.outdir, 'checkpoint.pth.tar')) 
        # Lr_scheduler
        if args.lr_decay:
            lr_scheduler.step()

def main():
    args = parse_args()
    train_loader, val_loader = prepare_data(args)
    args.num_class = 10 if args.dataset == 'cifar10' else 100
    model = models.__dict__[args.model](num_classes=args.num_class, expansion=args.expansion)
    args.Use_Cuda = torch.cuda.is_available()
    args.sum_channel = None
    if args.Use_Cuda:
        model.cuda()
    arch_train(model, args, train_loader, val_loader)
    new_model = prune(model, args)
    if args.Use_Cuda:
        new_model.cuda()
    weight_train(new_model, train_loader, val_loader, args)
    
if __name__ == '__main__':
    main()
import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
from prune import *
import argparse
from operator import itemgetter
from heapq import nsmallest
import time

class ModifiedVGG16Model(torch.nn.Module):
   def __init__(self):
        super(ModifiedVGG16Model, self).__init__()

        model = models.vgg16(pretrained=True)
        self.features = model.features

        for param in self.features.parameters():
            param.requires_grad = False


    def forward(self, x):
        x = self.features(x)
        return x

class FilterPruner:
    def __init__(self, model, device):
        self.model = model
        self.reset()
        self.device = device

    def reset(self):
        # self.activations = []
        # self.gradients = []
        # self.grad_index = 0
        # self.activation_to_layer = {}
        self.filter_ranks = {}

    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        activation_index = 0
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1

        #return self.model.classifier(x.view(x.size(0), -1))
        return x

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]
        values = \
            torch.sum((activation * grad), dim = 0).\
                sum(dim=-2).sum(dim=-1).data
        # Normalize the rank by the filter dimensions
        values = \
            values / (activation.size(0) * activation.size(2) * activation.size(3))

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = \
                torch.FloatTensor(activation.size(1)).zero_().to(self.device)

        self.filter_ranks[activation_index] += values
        self.grad_index += 1

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cpu()

    def get_pruning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

        # After each of the k filters are pruned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune

class PruningFineTuner_VGG16:
    def __init__(self, train_path, test_path, model, device):
        self.train_data_loader = dataset.loader(32,2)
        self.test_data_loader = dataset.test_loader(32,2)
        self.model = model
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.pruner = FilterPruner(self.model, device)
        self.model.train()

    def test(self):
        self.model.eval()
        correct = 0
        total = 0

        for i, (batch, label) in enumerate(self.test_data_loader):
            batch = batch.to(self.device)
            output = self.model(Variable(batch))
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label).sum()
            total += label.size(0)

        print("Accuracy :", float(correct) / total)

        self.model.train()

    def train(self, optimizer = None, epoches = 10):
        if optimizer is None:
            optimizer = \
                optim.SGD(self.model.parameters(),
                    lr=0.0001, momentum=0.9)

        for i in range(epoches):
            print("Epoch: ", i)
            self.train_epoch(optimizer)
            self.test()
        print("Finished fine tuning.")

    def train_batch(self, optimizer, batch, label, rank_filters):
        self.model.zero_grad()
        input = Variable(batch)

        if rank_filters:
            output = self.pruner.forward(input)
            self.criterion(output, Variable(label)).backward()
        else:
            self.criterion(self.model(input), Variable(label)).backward()
            optimizer.step()

    def train_epoch(self, optimizer = None, rank_filters = False):
        step= 0
        for batch, label in self.train_data_loader:
            print (label)
            self.train_batch(optimizer, batch.to(self.device), label.to(self.device), rank_filters)
            print("Step {}".format(step))
            step+=1


    def get_candidates_to_prune(self, num_filters_to_prune):
        self.pruner.reset()
        self.train_epoch(rank_filters = True)
        self.pruner.normalize_ranks_per_layer()

        return self.pruner.get_pruning_plan(num_filters_to_prune)

    def total_num_filters(self):
        filters = 0
        for name, module in self.model.features._modules.items():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                filters = filters + module.out_channels
        return filters

    def prune(self):
        #Get the accuracy before pruning
        self.test()

        self.model.train()

        #Make sure all the layers are trainable
        for param in self.model.features.parameters():
            param.requires_grad = True

        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = 512
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)

        iterations = int(iterations * 2.0 / 3)

        print("Number of pruning iterations to reduce 67% filters", iterations)

        for _ in range(iterations):
            print("Ranking filters.. ")
            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
            layers_pruned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_pruned:
                    layers_pruned[layer_index] = 0
                layers_pruned[layer_index] = layers_pruned[layer_index] + 1

            print("Layers that will be pruned", layers_pruned)
            print("Pruning filters.. ")
            model = self.model.cpu()
            for layer_index, filter_index in prune_targets:
                model = prune_vgg16_conv_layer(model, layer_index, filter_index, self.device)

            self.model = model.to(self.device)

            message = str(100*float(self.total_num_filters()) / number_of_filters) + "%"
            print("Filters pruned", str(message))
            self.test()
            print("Fine tuning to recover from pruning iteration.")
            optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            self.train(optimizer, epoches = 10)


        print("Finished. Going to fine tune the model a bit more")
        self.train(optimizer, epoches = 15)
        torch.save(model, "model_pruned")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--prune", dest="prune", action="store_true")
    parser.add_argument("--train_path", type = str, default = "./catdogs/train/")
    parser.add_argument("--test_path", type = str, default = "./catdogs/val/")
    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.train:
        model = ModifiedVGG16Model().to(device)
    elif args.prune:
        model = torch.load("model").to(device)

    fine_tuner = PruningFineTuner_VGG16(args.train_path, args.test_path, model, device)

    if args.train:
        fine_tuner.train(epoches = 5)
        torch.save(model, "model")

    elif args.prune:
        fine_tuner.prune()

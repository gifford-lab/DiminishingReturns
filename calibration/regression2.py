import numpy as np
import os
from matplotlib import pyplot as plt
from scipy import stats

from tqdm import tqdm
import pickle

import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

torch.cuda.set_device(7)

#preds, targets, weights (sorted by preds)
with open("./alldata_3_sources.pkl", 'rb') as fin:
    mhc1, mhc1a, mhc2a = pickle.load(fin)

# With scaled parameters (since matrix is lower triangular, i.e. conditioned)

class regressor(nn.Module):
    def __init__(self, data, windowSize):
        super(regressor, self).__init__()
        #dt = torch.double
        dt = torch.float
        
        _, targets, weights = data
        weights = weights/np.mean(weights)
        weightw = np.convolve(weights, np.ones(windowSize), mode = "valid")
        targetw = np.convolve(targets * weights, np.ones(windowSize), mode = "valid")/weightw
        
        #initval = torch.zeros((1,1,targets.size), dtype = dt)
        
        initval = np.concatenate( (np.zeros(windowSize), targetw) )
        initval = np.maximum.accumulate(initval , axis=0)
        
        normalize = 1/( np.arange(1, len(initval), 1)[::-1] )
        
        self.dx = nn.Parameter(torch.tensor((initval[1:] - initval[:-1])/normalize, dtype = dt), requires_grad=True)
        self.register_buffer("targetw", torch.tensor(targetw, dtype = dt))
        self.register_buffer("weightw", torch.tensor(weightw, dtype = dt))
        self.register_buffer("weights", torch.tensor(weights, dtype = dt).view(1,1,-1))
        self.register_buffer("convFilter", torch.ones((1,1,windowSize), dtype = dt))
        self.register_buffer("normalizer", torch.tensor(normalize, dtype = dt))
        
        dxFilter = np.ones(targets.size)
        dxFilter[:windowSize-1] = 0
        self.register_buffer("dxFilter", torch.tensor(dxFilter, dtype = dt))
        
        self.clip()
        
    def clip(self):
        with torch.no_grad():
            self.dx[:] = self.dx.clamp(min=0)
            
    def getModel(self):
        return (self.dx * self.normalizer).detach().cpu().numpy()
            
    def forward(self):
        xs = torch.cumsum(self.dx * self.dxFilter * self.normalizer, dim = 0)
        xs = nn.functional.conv1d(xs * self.weights, self.convFilter, padding = 0)
        xs = xs.view(-1) / self.weightw
        #return xs.detach().cpu().numpy(), self.targetw.cpu().numpy()
        return torch.sum( (xs - self.targetw) ** 2 )
        
def fitmodel(data, window, iterations):
    model = regressor(data, window)
    model = model.cuda()
    
    #optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr = 1)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr = 0.0000001)
    pbar = tqdm(range(iterations), position=0, leave=True)
    
    losses = []
    minloss = float('inf')
    bestmodel = None
    for i in pbar:
        optimizer.zero_grad()
        model.clip()
        loss = model()
        loss.backward()
        closs = loss.item()
        if closs < minloss:
            minloss = closs
            bestmodel = model.getModel()
        losses.append(closs)
        pbar.set_description("{:.8f}".format(losses[-1]))
        optimizer.step()
        
    model.clip()
    model = model.eval().cpu()
    return bestmodel, losses

#r = fitmodel(mhc1, 1000, 1000)
r1a = fitmodel(mhc1a, 1000, 100000)
r2a = fitmodel(mhc2a, 1000, 100000)
r1 = fitmodel(mhc1, 1000, 100000)

with open("./calicurve32_100k.pkl", 'wb') as fout:
    pickle.dump((r1, r1a, r2a), fout)
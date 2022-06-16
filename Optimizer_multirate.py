import math
import torch
import torch.nn as nn
from torch.optim import Optimizer

class Multirate(Optimizer):

    def __init__(self, params,lr=0.1,momentum=0,weight_decay=0):   
        defaults = dict(lr=lr, momentum=momentum,weight_decay=weight_decay)
        super(Multirate, self).__init__(params, defaults)
    
    @torch.no_grad()
    def initmom(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad

                param_state = self.state[p]
                buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                
    @torch.no_grad()
    def stepfast(self):

        for group in self.param_groups:
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                param_state = self.state[p]
                buf = param_state['momentum_buffer']
                
                if p.requires_grad:
                    d_p = p.grad
                    if weight_decay != 0:
                        d_p = d_p.add(p, alpha=weight_decay)
                    buf.mul_(momentum).add_(d_p)

                d_p = buf        
                p.add_(d_p, alpha=-group['lr'])

    @torch.no_grad()
    def stepslow(self):

        for group in self.param_groups:
            momentum = group['momentum']

            for p in group['params']:
                if p.requires_grad:
                    param_state = self.state[p]
                    buf = param_state['momentum_buffer']
                    
                    d_p = p.grad
                    if weight_decay != 0:
                        d_p = d_p.add(p, alpha=weight_decay)
                    buf.mul_(momentum).add_(d_p)



        

               

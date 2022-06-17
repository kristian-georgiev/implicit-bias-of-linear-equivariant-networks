import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi
from torch.nn.parameter import Parameter

def get_schatten_norm_sum(fcoeffs, irrep_sizes, l):
    """
    Given the Fourier coefficients of the linearization of our network,
    this f-n computes the expression for which we want to reach a scaling
    of its stationary point in our main theorems (Thm 4.2 and Thm 4.4).
    """
    
    # all we're doing here is getting the 2/l norm of the singular values
    # of the block diagonal matrix that has each of the irreps of dim n
    # repeated n times along its diagonal
    
    # our code could also be written more cleanly/concisely as
    # block_diag_matrix_irrs = torch.block_diag(*[torch.block_diag(*([flat_irrep.reshape(size, size)] * size)) 
    #                                                              for size, flat_irrep in zip(irrep_sizes, fcoeffs)])
    # sigmas = torch.linalg.svdvals(block_diag_matrix_irrs)
    # return torch.linalg.vector_norm(sigmas, ord=(2./3.))
    # but that'd be a bit slower because of the SVD on a larger (though sparse) matrix
    
    sum_of_schatten_norms = 0.
    flat_irreps_list = torch.split(fcoeffs, [s ** 2 for s in irrep_sizes])

    for flat_irrep, irrep_size in zip(flat_irreps_list, irrep_sizes):
        irrep = flat_irrep.reshape(irrep_size, irrep_size)
        sigmas = torch.linalg.svdvals(irrep)
        res = irrep_size * (torch.linalg.vector_norm(sigmas, ord=(2. / l)) ** (2. / l))
        sum_of_schatten_norms += res
    return sum_of_schatten_norms ** (l / 2.)



def f_mat_dh4():
    mat = np.asarray([  [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1,-1,-1,-1,-1],
                        [1,-1, 1,-1, 1,-1, 1,-1],
                        [1,-1, 1,-1,-1, 1,-1, 1],
                        [1, 0,-1, 0, 1, 0,-1, 0],
                        [0, 1, 0,-1, 0, 1, 0,-1],
                        [0,-1, 0, 1, 0, 1, 0,-1],
                        [1, 0,-1, 0,-1, 0, 1, 0],
                        ]).astype(np.complex)
    mat[:4,:] = mat[:4,:]/4*np.sqrt(2)
    mat[4:,:] = mat[4:,:]/2
    return mat.conj().T

def f_mat_unnorm():
    mat = np.asarray([  [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1,-1,-1,-1,-1],
                        [1,-1, 1,-1, 1,-1, 1,-1],
                        [1,-1, 1,-1,-1, 1,-1, 1],
                        [1, 0,-1, 0, 1, 0,-1, 0],
                        [0, 1, 0,-1, 0, 1, 0,-1],
                        [0,-1, 0, 1, 0, 1, 0,-1],
                        [1, 0,-1, 0,-1, 0, 1, 0],
                        ]).astype(np.complex)
    return mat


def get_dn_irreps(n=8):
    out = np.zeros((n,n)).astype(np.complex128)
    out[:2,:] = 1.
    out[1,n//2:] = -1.
    rots = np.arange(n//2)

    if n%4 ==0:
        a = np.arange(0,n,2)
        out[2,a] = 1
        out[2,a+1] = -1
        out[3,:] = 1
        out[3,a[:n//4]+1] = -1
        out[3,a[n//4:]] = -1
        left = (n-4)//4
        curr = 4
    else:
        left = (n-2)//4
        curr = 2
    for rho in range(left):
        out[rho*4+curr,:n//2] = np.cos(rots*pi*2/(n/2)) 
        out[rho*4+curr+1,:n//2] = -np.sin(rots*pi*2/(n/2)) 
        out[rho*4+curr+2,:n//2] = np.sin(rots*pi*2/(n/2)) 
        out[rho*4+curr+3,:n//2] = np.cos(rots*pi*2/(n/2))
        out[rho*4+curr+0,n//2:] = np.cos(rots*pi*2/(n/2)) 
        out[rho*4+curr+1,n//2:] = np.sin(rots*pi*2/(n/2)) 
        out[rho*4+curr+2,n//2:] = np.sin(rots*pi*2/(n/2)) 
        out[rho*4+curr+3,n//2:] = -np.cos(-rots*pi*2/(n/2)) 
    return out




def permutation_matrix(ids):
    mat = np.zeros((len(ids), len(ids))).astype(np.complex)
    for i, id_i in enumerate(ids):
        mat[id_i,i] = 1
    return mat

d4_ids = [
            [0,1,2,3,4,5,6,7],
            [1,2,3,0,5,6,7,4],
            [2,3,0,1,6,7,4,5],
            [3,0,1,2,7,4,5,6],
            [4,7,6,5,0,3,2,1],
            [5,4,7,6,1,0,3,2],
            [6,5,4,7,2,1,0,3],
            [7,6,5,4,3,2,1,0]]

per_mats = [permutation_matrix(ids)for ids in d4_ids]
reshape_ids = np.asarray(d4_ids) + np.arange(8).reshape(-1,1)*8
reshape_ids = reshape_ids.reshape(-1)
reshape_tens = torch.LongTensor(reshape_ids)

def d4_conv_funct(input,filter):
    start_shape = input.shape
    unfold_input = input.repeat(1,8)[:,reshape_tens]
    out = filter@torch.transpose(unfold_input.view(-1,8,8), -1,-2)	# may be more optimal with torch expand
    return torch.squeeze(out,1)


class d4_conv(nn.Module):
    def __init__(self,bias= False):
        super().__init__()
        self.in_features = 8
        self.out_features = 8
        self.weight = Parameter(torch.Tensor(1,self.in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        with torch.no_grad():
            self.weight /= 100.

    def forward(self, input):
        return d4_conv_funct(input, self.weight)

class d4_net(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([d4_conv() for i in range(n_layers)])
        # self.conv1 = d4_conv()
        # self.conv2 = d4_conv()
        self.fc1 = nn.Linear(8,1)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(x)
        return self.fc1(out)


class diagonal_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = Parameter(torch.Tensor(1,8))
        self.fc1 = nn.Linear(8,1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

    def forward(self,x):
        return self.fc1(self.w1*x)


def exp_loss(y_pred,y, normalize = False):
    if normalize:
        terms = torch.exp(-y_pred*y)*torch.exp(torch.abs(y))
    else:
        terms = torch.exp(-y_pred*y)
    return torch.sum(terms)

def get_linearization(net):
    return net(torch.eye(8))

if __name__ == '__main__':
    import torch.optim as optim
    torch.set_printoptions(sci_mode=False)

    A = get_dn_irreps(8)

    ins = torch.Tensor(f_mat_unnorm().conj().T[:,[2,3]].T)
    print(f"input: {ins}")
    outs = torch.Tensor([[1],[-1]])
    d4_mat = torch.Tensor(f_mat_dh4())
    n_epochs = 20000
    lr = 0.01
    normalize_loss = False

    net = d4_net(2)
    prior = (torch.transpose(d4_mat,0,1)@net(torch.eye(8).detach())).T 
    prior_r = (net(torch.eye(8).detach())).T 

    optimizer = optim.SGD(net.parameters(), lr=lr)
    losses = []
    schattens = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred = net(ins)
        loss = exp_loss(pred,outs, normalize_loss)
        loss.backward()
        optimizer.step()
        if (epoch%100)==0 and epoch <= 20000:
            lr *= 1.5
            optimizer = optim.SGD(net.parameters(), lr=lr)

        if (epoch%100) == 0:
            now_r = (net(torch.eye(8).detach())).T 
            now = (torch.transpose(d4_mat,0,1) @ net(torch.eye(8).detach())).T 
            prior = (torch.transpose(d4_mat,0,1) @ net(torch.eye(8).detach())).T 
            prior_r = (net(torch.eye(8).detach())).T 

        if epoch % 100 == 0:
            losses.append(loss.mean().cpu().item())
            beta = net(torch.eye(8))
            normalized_beta = beta / torch.linalg.vector_norm(beta)
            flattened_irreps = torch.tensor(f_mat_unnorm()).type(torch.complex64) @ normalized_beta.type(torch.complex64)
            sch_norm_sum = get_schatten_norm_sum(flattened_irreps, [1, 1, 1, 1, 2], 3)
            schattens.append(sch_norm_sum.detach().item())
        
    print(sch_norm_sum)
        
#     np.save('losses_3.npy', losses)
#     np.save('schattens_3.npy', schattens)
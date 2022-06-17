import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu
from torch.nn.parameter import Parameter
import torch.optim as optim
import torchvision
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
# from tqdm import tqdm
from tqdm.notebook import tqdm
import copy
import matplotlib as mpl
import pdb
import copy
mpl.rcParams['figure.autolayout'] = True

# tex_fonts = {
#     # Use LaTeX to write all text
#     "text.usetex": True,
#     "font.family": "serif",
#     # Use 10pt font in plots, to match 10pt font in document
#     "axes.labelsize": 10,
#     "font.size": 10,
#     # Make the legend/label fonts a little smaller
#     "legend.fontsize": 8,
#     "xtick.labelsize": 8,
#     "ytick.labelsize": 8
# }
# plt.rcParams.update(tex_fonts)

# mpl.rc('font', **{'family' : "sans-serif"})

#params= {'text.latex.preamble' : [r'\usepackage{amsmath}']}
#plt.rcParams.update(params)

# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.style.use('seaborn')

def permutation_matrix(ids):
    """
    Go from permutation of [N] to an NxN permutation matrix.
    """
    mat = np.zeros((len(ids), len(ids)))
    for i, id_i in enumerate(ids):
        mat[id_i, i] = 1
    return mat


class Group():
    """
    A small class that exposes group properties we'll need - the order of the group,
    the dimensions of its irreducible representations, its Fourier matrix, Cayley table,
    and lastly, a transformation (self.tensorize_inds) that transforms the inputs of a
    neural network according to the formulation of the data tensor in Yun et al., 2020,
    https://arxiv.org/abs/2010.02501.
    """
    def __init__(self, name):
        self.name = name
        self.order = self.get_order(self.name)
        self.irrep_sizes = self.get_irrep_sizes(self.name)
        
        f_mat = torch.from_numpy(np.load(f'data/unscaled_bases/{self.name}.npy')).type(torch.complex64)
        self.f_mat = torch.conj(f_mat).T
        
        # we instead use non-normalized DFT matrices
#         assert torch.allclose((self.f_mat @ torch.conj(self.f_mat).T).real, torch.eye(self.order), atol=1e-6), 'group DFT matrix must be unitary (real part does not match identity)'
#         assert torch.allclose((self.f_mat @ torch.conj(self.f_mat).T).imag, torch.zeros((self.order, self.order)), atol=1e-6), 'group DFT matrix must be unitary (imag part does not match zero)'

        self.cayley = np.load(f'data/cayley/{self.name}_Cayley.npy')
        reshape_ids = np.asarray(self.cayley) + np.arange(self.order).reshape(-1, 1) * self.order
        reshape_ids = reshape_ids.reshape(-1)
        self.tensorize_inds = torch.LongTensor(reshape_ids)
        
    def get_order(self, name):
        name_to_order = {'D8': 8,
                         'D60': 60,
                         'A5': 60,
                         'C10C10C2': 200,
                         'CCQ200': 200,
                         'MNIST_toy': 72,
                         'MNIST': 6272}
        return name_to_order[name]
    
    def get_irrep_sizes(self, name):
        group_to_irrep_sizes = {'D8': (1, 1, 1, 1, 2),
                                'D60': (1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2),
                                'A5': (1, 3, 3, 4, 5),
                                'C10C10C2': tuple([1] * 200),
                                'CCQ200': (1, 1, 1, 1, 2, 8, 8, 8),
                                'MNIST_toy': tuple([1] * 36 + [2] * 9),
                                'MNIST': tuple([1] * 3136 + [2] * 784)}
        return group_to_irrep_sizes[name]


def exp_loss(y_pred, y):
    """
    Exponential loss for classification.
    """
    return torch.mean(torch.exp(-y_pred * y))

def g_conv_func(inp: torch.Tensor,
                filt: torch.Tensor,
                group_order: int,
                group_tensorize_inds: torch.Tensor):
    """
    Implements a G-convolution for a given group G.
    """
    n = group_order
    # inp has size (batch, n)
    unfold_input = (inp.repeat(1, n)[:, group_tensorize_inds]).reshape(-1, n, n)
    # unfolded input has size (batch, n, n)
    unfold_input = torch.transpose(unfold_input, -1, -2)
    # filt has size (1, n)
    out = filt @ unfold_input
    return torch.squeeze(out, 1)

class g_conv(nn.Module):
    """
    torch.nn.Conv1D-style conv module for group convolutions.
    """
    def __init__(self, group, bias=False, nonlinear=False):
        super().__init__()
        self.group = group
        self.nonlinear = nonlinear
        self.n = self.group.order
        self.in_features = self.n
        self.out_features = self.n
        self.weight = Parameter(torch.Tensor(1, self.in_features))
        self.reset_parameters()


    def reset_parameters(self):
        nonlinearity = 'linear' if not self.nonlinear else 'relu'
        nn.init.kaiming_uniform_(self.weight, nonlinearity=nonlinearity)
        # with torch.no_grad():
        #     self.weight *= np.sqrt(self.n)

    def forward(self, input):
        return g_conv_func(inp=input, filt=self.weight,
                           group_order=self.group.order,
                           group_tensorize_inds=self.group.tensorize_inds)


class circular_conv(nn.Module):
    """
    torch.nn.Conv1D-style conv module for circular convolutions.
    """
    def __init__(self, n, bias=False, nonlinear=False):
        super().__init__()
        self.n = n
        self.nonlinear = nonlinear
        self.in_features = self.n
        self.out_features = self.n
        self.pad_ln = n // 2

        self.weight = torch.Tensor(1, self.n)
        self.reset_parameters()
        self.weight = Parameter(self.weight)

    def reset_parameters(self):
        nonlinearity = 'linear' if not self.nonlinear else 'relu'
        nn.init.kaiming_uniform_(self.weight, nonlinearity=nonlinearity)
        # with torch.no_grad():
        #     self.weight /= np.sqrt(self.n)

    def forward(self, inp):
        # circular padding in torch is only available for 3D and 4D tensors (for some reason)
        # so we add a 'fake' dimension to carry out the padding, and squeeze it back afterwards.
        padded_weights = F.pad(self.weight.reshape([1, 1, -1]), pad=[self.pad_ln,
                                                                     self.pad_ln - 1], mode='circular')
        result = F.conv1d(padded_weights, inp.unsqueeze(1))
        return torch.squeeze(result, dim=0)



# =======================
# === Linear networks ===
# =======================


class g_net(nn.Module):
    """
    A simple implementation of a linear G-CNN.
    """
    def __init__(self, group):
        """
        n: (int) group order
        """
        super().__init__()
        self.group = group
        self.n = self.group.order
        self.nonlinear = False
        self.is_g = True

        self.conv1 = g_conv(self.group, nonlinear=self.nonlinear)
        self.conv2 = g_conv(self.group, nonlinear=self.nonlinear)
        self.lin_layer = torch.Tensor(self.n, 1)
        self.reset_parameters()
        self.fc1 = Parameter(self.lin_layer)

        self.l = 3  # this includes the linear layer fc1

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        stdv = 1. / math.sqrt(self.n)
        self.lin_layer.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return (self.conv2(self.conv1(x))) @ self.fc1
        

class conv_net(nn.Module):
    """
    A simple linear convolutional network with circular convolutions.
    """
    def __init__(self, group):
        super().__init__()
        self.group = group
        self.n = self.group.order
        self.nonlinear = False
        self.is_g = False

        self.conv1 = circular_conv(self.n, nonlinear=self.nonlinear)
        self.conv2 = circular_conv(self.n, nonlinear=self.nonlinear)
        self.lin_layer = torch.Tensor(self.n, 1)
        
        self.reset_parameters()
        self.fc1 = Parameter(self.lin_layer)

        self.l = 3

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        stdv = 1. / math.sqrt(self.n)
        self.lin_layer.data.uniform_(-stdv, stdv)

    def forward(self, x):
        ret = self.conv2(self.conv1(x))
        return (ret @ self.fc1)


class fc_net(nn.Module):
    """
    A simple linear fully-connected network.
    """
    def __init__(self, group):
        super().__init__()
        self.group = group
        self.n = self.group.order
        
        self.lin_layer1 = torch.Tensor(self.n, self.n)
        self.lin_layer2 = torch.Tensor(self.n, self.n)
        self.lin_layer3 = torch.Tensor(self.n, 1)

        self.reset_parameters()
        self.fc1 = Parameter(self.lin_layer1)
        self.fc2 = Parameter(self.lin_layer2)
        self.fc3 = Parameter(self.lin_layer3)

        self.nonlinear = False
        self.is_g = False
        self.l = 3

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.n)
        self.lin_layer1.data.uniform_(-stdv, stdv)
        self.lin_layer2.data.uniform_(-stdv, stdv)
        self.lin_layer3.data.uniform_(-stdv, stdv)

    def forward(self,x):
        return x @ self.fc1 @ self.fc2 @ self.fc3


# ===========================
# === Non-linear networks ===
# ===========================


def real_relu(z):
    # inputs and outputs are reals anyway, only use complex numbers to make it easier with DFT matrices (which can be complex)
    if torch.is_complex(z):
        return relu(z.real) + 1.j * 0.
    else:
        return relu(z)


class relu_g_net(nn.Module):
    def __init__(self, group):
        """
        n: (int) group order
        """
        super().__init__()
        self.group = group
        self.n = self.group.order
        self.conv1 = g_conv(self.group)
        self.conv2 = g_conv(self.group)
        self.lin_layer = torch.Tensor(self.n, 1)
        self.reset_parameters()
        self.fc1 = Parameter(self.lin_layer)
        self.relu = real_relu
        
        self.nonlinear = True
        self.is_g = True
        self.l = 3

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        stdv = 1. / math.sqrt(self.n)
        self.lin_layer.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x)))) @ self.fc1


class relu_conv_net(nn.Module):
    def __init__(self, group):
        super().__init__()
        self.group = group
        self.n = self.group.order

        self.conv1 = circular_conv(self.n)
        self.conv2 = circular_conv(self.n)
        self.lin_layer = torch.Tensor(self.n, 1)
        
        self.reset_parameters()
        self.fc1 = Parameter(self.lin_layer)
        self.relu = real_relu

        self.nonlinear = True
        self.is_g = False
        self.l = 3

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        stdv = 1. / math.sqrt(self.n)
        self.lin_layer.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)        
        return self.relu(self.conv2(x)) @ self.fc1

class relu_fc_net(nn.Module):
    def __init__(self, group):
        super().__init__()
        self.group = group
        self.n = self.group.order
        
        self.lin_layer1 = torch.Tensor(self.n, self.n)
        self.lin_layer2 = torch.Tensor(self.n, self.n)
        self.lin_layer3 = torch.Tensor(self.n, 1)

        self.reset_parameters()
        self.fc1 = Parameter(self.lin_layer1)
        self.fc2 = Parameter(self.lin_layer2)
        self.fc3 = Parameter(self.lin_layer3)
        self.relu = real_relu

        self.nonlinear = False
        self.is_g = False
        self.l = 3

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.n)
        self.lin_layer1.data.uniform_(-stdv, stdv)
        self.lin_layer2.data.uniform_(-stdv, stdv)
        self.lin_layer3.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return self.relu(self.relu(x @ self.fc1) @ self.fc2) @ self.fc3


def complex_pooling(x):
    return torch.max(x.real, dim=-1, keepdim=True)[0]

class relu_g_net_pool(nn.Module):
    def __init__(self, group):
        """
        n: (int) group order
        """
        super().__init__()
        self.group = group
        self.n = self.group.order
        self.conv1 = g_conv(self.group)
        self.conv2 = g_conv(self.group)
        self.pooling_layer = torch.nn.AvgPool1d(group.order)
        self.reset_parameters()
        self.relu = real_relu
        
        self.nonlinear = True
        self.is_g = True
        self.l = 3

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x):
        x = self.relu(self.conv2(self.relu(self.conv1(x))))
        return self.pooling_layer(x.unsqueeze(1)).squeeze(1)


class relu_conv_net_pool(nn.Module):
    def __init__(self, group):
        super().__init__()
        self.group = group
        self.n = self.group.order

        self.conv1 = circular_conv(self.n)
        self.conv2 = circular_conv(self.n)
        self.pooling_layer = torch.nn.AvgPool1d(group.order)
        
        self.reset_parameters()
        self.relu = real_relu

        self.nonlinear = True
        self.is_g = False
        self.l = 3

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
#         stdv = 1. / math.sqrt(self.n)

    def forward(self, x):
        # x = self.relu(self.conv2(self.relu(self.conv1(x))))
        x = self.conv2(self.relu(self.conv1(x)))
        return self.pooling_layer(x.unsqueeze(1)).squeeze(1)

class relu_fc_net_pool(nn.Module):
    def __init__(self, group):
        super().__init__()
        self.group = group
        self.n = self.group.order
        
        self.lin_layer1 = torch.Tensor(self.n, self.n)
        self.lin_layer2 = torch.Tensor(self.n, self.n)

        self.reset_parameters()
        self.fc1 = Parameter(self.lin_layer1)
        self.fc2 = Parameter(self.lin_layer2)
        self.relu = real_relu
        self.pooling_layer = torch.nn.AvgPool1d(group.order)

        self.nonlinear = False
        self.is_g = False
        self.l = 3

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.n)
        self.lin_layer1.data.uniform_(-stdv, stdv)
        self.lin_layer2.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = self.relu(self.relu(x @ self.fc1) @ self.fc2)
        return self.pooling_layer(x.unsqueeze(1)).squeeze(1)

        
    
def linearize_net_around_inp(net, inp):
    """
    we do not use biases, so no need to shift to pass through net(inp)
    """
    inp_temp = Parameter(inp.detach(), requires_grad=True)
    outp = net(inp_temp)
    outp.sum().backward()  # same effect as getting gradient per output
    return inp_temp.grad

def estimate_schatten_norm_of_nonlinear_net(net_state_dict, net_class, ins, group, mode='fouirier', l=3, cuda=False):
    schattens = []
    net = net_class(group)
    net.load_state_dict(net_state_dict)
    if cuda:
        net = net.cuda()
    linearizations = linearize_net_around_inp(net, ins)
    for linearization in linearizations:
        s, _ = net_schatten_norm(linearization, group, mode=mode, net_l=l, cuda=cuda)
        schattens.append(s)
    try:
        return torch.mean(torch.stack(schattens))
    except:
        return torch.zeros([])


# =======================================
# === Training, plotting, and helpers ===
# =======================================

def get_linearization(net, group, cuda=False):    
    if torch.is_tensor(net):
        return net.reshape(-1, 1)
    elif net.nonlinear:
        # will linearize around each point later on
        net = net.cpu()
        sd = copy.deepcopy(net.state_dict())
        if cuda:
            net = net.cuda()
        return sd
    elif cuda:
        net = net.cuda()
        lin = [net(shard.cuda()).detach()
               for shard in torch.eye(group.order).split(15)]
        lin = torch.cat(lin)
        return lin
    else:
        lin = [net(elt.view(1, -1)).detach()
               for elt in torch.eye(group.order)]
        lin = torch.stack(lin).reshape(-1, 1)
        return lin


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
        try:
            sigmas = torch.linalg.svdvals(irrep)
        except:
            return None
        sum_of_schatten_norms += irrep_size * torch.linalg.vector_norm(sigmas, ord=(2. / l)) ** (2. / l)
    return sum_of_schatten_norms ** (l / 2.)


def net_schatten_norm(beta, group, mode='fourier', net_l=3, cuda=False):
    """
    Given a linearization of a network, together with a group, computes the expression
    for which we want to reach a scaling of its stationary point in our main
    theorems (Thm 4.2 and Thm 4.4).
    """
    normalized_beta = beta / torch.linalg.vector_norm(beta)

    if mode == "fourier":
        flattened_irreps = (torch.conj(group.f_mat).T @ normalized_beta.type(torch.complex64)).detach()
    elif mode == "real":
        return torch.linalg.vector_norm(normalized_beta.data.reshape(-1),
                                        ord=(2. / net_l)), group.order

    else:
        raise ValueError('mode must be real or Fourier')

    sch_norm_sum = get_schatten_norm_sum(flattened_irreps,
                                         group.irrep_sizes,
                                         net_l)
    return sch_norm_sum, flattened_irreps


def train_net(net, group, dataloader, postprocess_fn, n_epochs, init_lr, cuda):
    if cuda:
        net = net.cuda()
        group.f_mat = group.f_mat.cuda()
        group.tensorize_inds = group.tensorize_inds.cuda()

    optimizer = optim.SGD(net.parameters(), lr=init_lr)
    linearized_nets = []
    schatten_norms_over_time = []
    real_schatten_norms_over_time = []
    loss_over_time = []
    acc_over_time = []

    # itr = range(n_epochs)
    itr = tqdm(range(n_epochs))
    for epoch in itr:
        if (epoch + 1) % 100 == 0:
            for g in optimizer.param_groups:
                g['lr'] *= 1.1

        loss_epoch = 0.
        with torch.no_grad():
            linearized_nets.append(get_linearization(net, group, cuda))
        #pdb.set_trace()
        for i, batch in enumerate(dataloader):
            if i == 2:
                break # use only part of the train set
            try:
                inputs, outputs = postprocess_fn(batch)
            except:
                continue
            if cuda:
                inputs = inputs.cuda()
                outputs = outputs.cuda()                
            optimizer.zero_grad()
            preds = net(inputs)
            # if epoch == n_epochs - 1:
            #     print(f'preds: {preds}')
            loss = exp_loss(preds, outputs)
            loss.backward()
            # print("you better not be imaginary ", list(net.parameters())[0].grad)
            
            optimizer.step()
            loss_epoch += loss.item().real

        loss_over_time.append(loss_epoch)
        if math.isnan(loss_over_time[-1]):
            print('try again')
            return [], [float('nan')], []  # start over

        acc_over_time.append(torch.mean((torch.sign(preds).data == outputs).type(torch.float32)).item())

    return linearized_nets, loss_over_time, acc_over_time, net

def get_training_trajectories(net_class, N, group, dataloader, postprocess_fn, epochs, lr, cuda,
                              experiment_name):
    """
    Train a network N times and report the trajectories throughout training of the sum of schatten norms over irreps
    of the (group) Fourier coefficients of the linearization of the net, and losses. Provide trajectories as pandas
    dataframes.
    """
    trajs = []
    times_nan = -N
    for _ in tqdm(range(N)):
        loss = float('nan')
        while math.isnan(loss) or loss > 0.1:
            net = net_class(group)
            linearized_nets, loss_over_time, acc_over_time, net = train_net(net, group,
                                                                            dataloader=dataloader,
                                                                       postprocess_fn=postprocess_fn,
                                                                       n_epochs=epochs,
                                                                       init_lr=lr,
                                                                       cuda=cuda)
            loss = loss_over_time[-1]
            times_nan += 1

        schatten_norms_over_time = []
        real_schatten_norms_over_time = []
        for beta in linearized_nets:  # calculate Schatten norms over the course of training (slow, so putting it outside of training which diverges sometimes)
            if net.nonlinear:
                batch = next(iter(dataloader))  # get just a single batch to evaluate on
                ins, _ = postprocess_fn(batch)
                if cuda:
                    ins = ins.cuda()
                # beta is a copy of the network in this cse, will linearize around each input within f-n below
                sch_norm_sum = estimate_schatten_norm_of_nonlinear_net(beta, net_class,
                                                                       ins, group, mode='fourier',
                                                                       cuda=cuda)
                r_sch_norm_sum = estimate_schatten_norm_of_nonlinear_net(beta, net_class,
                                                                         ins, group, mode='real',
                                                                         cuda=cuda)
            else:
                with torch.inference_mode():
                    sch_norm_sum, _ = net_schatten_norm(beta, group, 'fourier', cuda=cuda)
                    r_sch_norm_sum, _ = net_schatten_norm(beta, group, 'real', cuda=cuda)

            schatten_norms_over_time.append(sch_norm_sum)
            real_schatten_norms_over_time.append(r_sch_norm_sum)
        
        trajs.append((schatten_norms_over_time, loss_over_time, real_schatten_norms_over_time, acc_over_time))

    print(f"{net_class} did not converge {times_nan} times to get {N} successes.")
    all_sch_norms = np.array([np.array(list(zip([x.cpu() for x in traj_list[0]], range(len(traj_list[0]))))) for traj_list in trajs]).T.reshape(2, -1).T
    df_sch = pd.DataFrame(all_sch_norms, columns=['schatten', 'epoch'])

    losses = np.array([np.array(list(zip(traj_list[1], range(len(traj_list[1]))))) for traj_list in trajs]).T.reshape(2, -1).T
    df_losses = pd.DataFrame(losses, columns=['loss', 'epoch']).clip(lower=0.)  # ignore numerical instabilities
    df_losses = df_losses[df_losses['loss'] < 10]  # ignore annoying spikes at the very start that distort the yrange

    allr_sch_norms = np.array([np.array(list(zip([x.cpu() for x in traj_list[2]], range(len(traj_list[2]))))) for traj_list in trajs]).T.reshape(2, -1).T
    df_r_sch = pd.DataFrame(allr_sch_norms, columns=['r_schatten', 'epoch'])

    return (df_sch, df_losses, df_r_sch) , net


def plot_schatten_norm_sums_and_loss(nets, group, dataloader, postprocess_fn, N, relu=False, epochs=1500, lr=0.01, cuda=False, dfs={}, exp_name="experiment", horizontal_lb=0):
    """
    A high-level function used to train G-CNNs, CNNs, and FC nets, and plot
    the expressions of interest (sum of schatten norms of irreps and loss)
    for each of the networks throughout the course of training.
    """
    color_palette = sns.color_palette('colorblind', 3)
    scaling = 1.8
    # plt.rcParams["text.usetex"] = True

    if len(dfs) == 0:
        for net_name in nets.keys():
            dfs[net_name], net = get_training_trajectories(nets[net_name], N, group, dataloader,
                                                           postprocess_fn, epochs, lr, cuda,
                                                           exp_name)
    else:
        net = None

    plt.figure(figsize=(scaling * 2.695, scaling * 1.666))
    for i, net_name in enumerate(dfs.keys()):
        df_sch, df_losses, _ = dfs[net_name]
        lp = sns.lineplot(data=df_sch, x="epoch", y="schatten", label=net_name, color=color_palette[i], ci=None)
    lp.axhline(horizontal_lb, linestyle='--', xmax=400, label='optimal')
    plt.xlabel('epoch')
    # plt.ylabel(r'$\displaystyle\|\widehat{\boldsymbol{\beta}}\|^{(S)}_{2/L}$')
    plt.ylabel('schatten norm')
    plt.legend(loc=(0.75, 0.20))
    fig = plt.gcf()
    # fig.savefig(f'figures/{exp_name}_fourier_space.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    
    plt.figure(figsize=(scaling * 2.695, scaling * 1.666))
    for i, net_name in enumerate(dfs.keys()):
        _, df_losses, df_r_sch = dfs[net_name]
        sns.lineplot(data=df_r_sch, x="epoch", y="r_schatten", label=net_name, color=color_palette[i], ci=None)
    plt.xlabel('epoch')
    # plt.ylabel(r'$\left\vert\left\vert \boldsymbol{\beta}\right\vert\right\vert_{2/L}$')
    plt.ylabel('real norm')
    plt.legend(loc='best')
    fig = plt.gcf()
    # fig.savefig(f'figures/{exp_name}_real_space.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(scaling * 2.695, scaling * 1.666))
    for i, net_name in enumerate(dfs.keys()):
        _, df_losses, df_r_sch = dfs[net_name]
        sns.lineplot(data=df_losses, x="epoch", y="loss", label=net_name, color=color_palette[i], ci=None)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    fig = plt.gcf()
    # fig.savefig(f'figures/{exp_name}_loss.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
    return dfs, net
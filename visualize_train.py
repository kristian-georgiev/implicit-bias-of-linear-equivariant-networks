import torch
import numpy as np
import pickle
from matplotlib import pyplot as plt
import matplotlib as mpl
# local imports
from src.nn_training import Group, plot_schatten_norm_sums_and_loss, train_net
from src.nn_training import g_net, conv_net, fc_net, relu_g_net, relu_conv_net, relu_fc_net

mpl.rcParams['text.usetex'] = False
mpl.rc('font', **{	'family' : 'normal',
        			'size'   : 10})




def visualize_outputs(nets, real_start, real_end, fourier_start, fourier_end, exp_name="experiment"):
	"""
	A high-level function used to train G-CNNs, CNNs, and FC nets, and plot
	the expressions of interest (sum of schatten norms of irreps and loss)
	for each of the networks throughout the course of training.
	"""

	# plt.figure(figsize=(scaling * 2.695, scaling * 1.666))
	def format_axis(ax):
		# ax.set_xticks(np.arange(-.5, 60, 1), minor=True)
		# ax.set_yticks(np.arange(-.5, 1, 1), minor=True)
		# ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.spines['left'].set_visible(False)
		ax.spines['bottom'].set_visible(False)

	def format_data(data):
		data = torch.abs(data)
		data = data / torch.max(data)
		return data.cpu().numpy().reshape(1,-1)

	fig, axs = plt.subplots(nrows=3*3+1,ncols=3, figsize=(7, 3))
	linewidth = 0.25
	vmax = 0.8
	vmin = 0.
	# axs[-1,0].axis('off')
	axs[-1,1].axis('off')
	axs[-1,2].axis('off')
	axs[-1,1].text(0.5,0.5, 'Real regime coefficients', horizontalalignment = 'center')
	axs[-1,2].text(0.5,0.5, 'Fourier regime coefficients', horizontalalignment = 'center')

	for i, net_name in enumerate(nets.keys()):
		ex_plot = axs[3*i,1].pcolormesh(format_data(real_start[net_name]), cmap='Purples', edgecolors = 'gray', linewidth = linewidth, vmax = vmax, vmin = vmin)
		axs[3*i,2].pcolormesh(format_data(fourier_start[net_name]), cmap = 'Purples', edgecolors = 'gray', linewidth = linewidth, vmax = vmax, vmin = vmin)
		axs[3*i+1,1].pcolormesh(format_data(real_end[net_name]), cmap='Purples', edgecolors = 'gray', linewidth = linewidth, vmax = vmax, vmin = vmin)
		axs[3*i+1,2].pcolormesh(format_data(fourier_end[net_name]), cmap='Purples', edgecolors = 'gray', linewidth = linewidth, vmax = vmax, vmin = vmin)
		
		format_axis(axs[3*i+0,1])
		format_axis(axs[3*i+1,1])
		format_axis(axs[3*i+0,2])
		format_axis(axs[3*i+1,2])
		axs[3*i+2,1].remove()
		axs[3*i+2,2].remove()
		axs[3*i+0,0].axis('off')
		axs[3*i+1,0].axis('off')
		axs[3*i+2,0].remove()

		axs[3*i,0].text(1.,0.5, net_name + ' (at initialization):', horizontalalignment = 'right')
		axs[3*i+1,0].text(1.,0.5, net_name + ' (trained):', horizontalalignment = 'right')

	# plt.ylabel(r'$\displaystyle\|\widehat{\boldsymbol{\beta}}\|^{(S)}_{2/L}$')

	cmap = mpl.cm.Purples
	norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
	cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
						orientation = 'horizontal', 
						cax=axs[-1,0],
						ticks = [0,0.2,0.4,0.6,0.8])
	cbar.ax.tick_params(labelsize=8)
	cbar.outline.set_edgecolor('black')
	cbar.set_label(label='magnitude (relative to maximum entry)',size=8)

	plt.subplots_adjust(wspace=0, hspace=0)
	fig = plt.gcf()
	fig.savefig(f'figures/{exp_name}_fourier_space.pdf', format='pdf', bbox_inches='tight')



def transform_linearization(beta, group, mode='fourier', cuda=False):
	"""
	Given a linearization of a network, together with a group, computes the expression
	for which we want to reach a scaling of its stationary point in our main
	theorems (Thm 4.2 and Thm 4.4).
	"""
	normalized_beta = beta / torch.linalg.vector_norm(beta)

	if mode == "fourier":
		return (torch.conj(group.f_mat).T @ normalized_beta).detach()
	elif mode == "real":
		return normalized_beta
	else:
		raise ValueError('mode must be real or Fourier')



def get_training_dataframes(experiment_name, force_train):
	if not force_train:
		try:
			with open(f'data/training/{experiment_name}.pickle', 'rb') as f:
				dfs = pickle.load(f)
		except:
			print('Will train before plotting.')
			dfs = {}  # if training has not been done yet
	else:
			print('Will train before plotting.')
			dfs = {}
	return dfs

experiment_name = "d60_gaussian_10_sep_visuals"
group = Group('D60')
k = 5
ins_m = torch.complex(torch.randn([k, 60]) , torch.zeros([k, 60]))
ins_p = torch.complex(torch.randn([k, 60]) , torch.zeros([k, 60]))
ins = torch.cat([ins_m, ins_p])
outs_p = torch.Tensor([[-1]] * k)
outs_m = torch.Tensor([[1]] * k)
outs = torch.cat([outs_m, outs_p])

# N = 30 # average over trajectories
# force_train = False

nets = {"G-CNN": g_net, "CNN": conv_net, "FC": fc_net}
start_linearizations = {}
end_linearizations = {}


for net_name in nets:
	net = nets[net_name](group)
	print(net)
	linearized_nets, loss_over_time, acc_over_time = train_net(net, group,
																   inputs=ins,
																   outputs=outs,
																   n_epochs = 1500,
																   init_lr = 0.1,
																   cuda=False)

	start = linearized_nets[0]
	end = linearized_nets[-1]
	start_linearizations[net_name] = start.detach()
	end_linearizations[net_name] = end.detach()

fourier_start_linearizations = {}
fourier_end_linearizations = {}
for net_name in nets:
	fourier_start_linearizations[net_name] = transform_linearization(start_linearizations[net_name], group)
	fourier_end_linearizations[net_name] = transform_linearization(end_linearizations[net_name], group)

print(torch.abs(fourier_end_linearizations['G-CNN']))
print(torch.abs(fourier_end_linearizations['FC']))

visualize_outputs(nets, start_linearizations, end_linearizations, fourier_start_linearizations, fourier_end_linearizations, exp_name=experiment_name)

# dfs = get_training_dataframes(experiment_name, force_train)

# dfs = plot_schatten_norm_sums_and_loss(nets, group, ins, outs, N, epochs=100, cuda=False, dfs=dfs, exp_name=experiment_name)
# with open(f'data/training/{experiment_name}.pickle', 'wb') as f:
#     pickle.dump(dfs, f, protocol=pickle.HIGHEST_PROTOCOL)
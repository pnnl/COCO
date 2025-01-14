import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from matplotlib.lines import Line2D

import neuromancer.psl as psl
from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuromancer.plot import pltCL

from matplotlib.lines import Line2D

from cvxpylayers.torch.cvxpylayer import CvxpyLayer
import cvxpy as cv

from building_MPC_layer import get_building_MPC_layer, get_building_MPC_layer
import matplotlib.pyplot as plt
import Nets
from building_correction import B_correction

import os
import building_utils
import pickle
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--ntrain',     type=int,     default=10000)
parser.add_argument('--ntest',      type=int,     default=1000)
parser.add_argument('--lr',         type=float,   default=1e-3)
parser.add_argument('--nsteps',     type=int,     default=50)
parser.add_argument('--nzones',     type=int,     default=1)
parser.add_argument('--penalty',    type=float,   default=1000.0)
parser.add_argument('--alpha',      type=float,   default=1e-5)
parser.add_argument('--index',      type=int,     default=9999)

parser.add_argument('--upos',       type=bool,    default=False)
parser.add_argument('--epochs',     type=int,     default=10)
args = parser.parse_args()



torch.manual_seed(0)
np.random.seed(0)

# ground truth system model
sys = psl.systems['LinearSimpleSingleZone'](seed=0)
nzones = args.nzones

# problem dimensions
nx = nzones*sys.nx                # number of states
nu = nzones*sys.nu                # number of control inputs
nd = nzones*sys.nD                # number of disturbances
ny = nzones*sys.ny                # number of controlled outputs
nref = ny                  # number of references
partial_observe = False
if partial_observe:        # Toggle partial observability of the disturbance
    d_idx = sys.d_idx
else:
    d_idx = range(nd)
nd_obs = len(d_idx)   # number of observable disturbances
nB = nu*nx  # This is the size of B as passed in to the policy network, should be the number of learnable elements

# extract exact state space model matrices:
A = torch.block_diag(  *tuple([torch.tensor(sys.A)    for _ in range(nzones)])  )
B = torch.block_diag(  *tuple([torch.tensor(sys.Beta) for _ in range(nzones)])  )
C = torch.block_diag(  *tuple([torch.tensor(sys.C)    for _ in range(nzones)])  )
E = torch.block_diag(  *tuple([torch.tensor(sys.E)    for _ in range(nzones)])  )
F = torch.zeros(ny)
G = torch.zeros(nx)
y_ss = torch.zeros(ny)

print("A = ")
print( A    )
print("A.sum(0) = ")
print( A.sum(0)    )
print("A.sum(1) = ")
print( A.sum(1)    )
print("torch.tensor(sys.E) = ")
print( torch.tensor(sys.E)    )
dx = 0.05
nx_zone = 4
if False:#nzones > 1:
    A = building_utils.link_zones_long(A, nzones, nx_zone, dx)

# control action bounds
#umin = torch.Tensor([sys.umin.item() for _ in range(nzones)])
umax = torch.Tensor([sys.umax.item() for _ in range(nzones)])

B_scale_factor = umax.max()
umax = umax / B_scale_factor    # JK rescaling reduces computation time dramatically
B = B * B_scale_factor          #   however, I haven't been able to verify equivalence, due to either solver instability when R!=0 or nonunique solutions when R=0

umin = torch.Tensor([sys.umin.item() for _ in range(nzones)]) if args.upos else -umax


print("umin")
print( umin )

nsteps = args.nsteps
ntrain = args.ntrain
ntest  = args.ntest
batch_size_ul = 100

n_samples = ntrain+ntest


print("ntrain")
print( ntrain )
print("nsteps")
print( nsteps )
print("nzones")
print( nzones )


"""
Generate the main data defining each control task:
bounds ymin, ymax;
disturbances dist;
initial state x0
"""
# generate data for a single-zone building
filename = "./data/zone_data_{}steps_{}samples.p".format(nsteps, n_samples)
if not os.path.isfile(filename):
    zone_data = building_utils.gen_building_data_single(n_samples, nsteps)
    pickle.dump(zone_data,open(filename,'wb'))
else:
    zone_data = pickle.load(open(filename,'rb'))


batched_ymin, batched_ymax, batched_dist, batched_x0 = zone_data

# duplicate data across multiple zones
batched_x0   = batched_x0.repeat(1,1,nzones)
batched_y0   = batched_x0[:,:,[-1]].repeat(1,1,nzones)
batched_ymin = batched_ymin.repeat(1,1,nzones)
batched_ymax = batched_ymax.repeat(1,1,nzones)
batched_dist = batched_dist.repeat(1,1,nzones)




"""
Train/test split
"""
train_ymin = batched_ymin[:ntrain]
train_ymax = batched_ymax[:ntrain]
train_dist = batched_dist[:ntrain]
train_x0   =   batched_x0[:ntrain]

test_ymin = batched_ymin[-ntest:]
test_ymax = batched_ymax[-ntest:]
test_dist = batched_dist[-ntest:]
test_x0   =   batched_x0[-ntest:]


"""
Upper-level prediction model components
"""
relu = torch.nn.ReLU()
sigmoid = torch.nn.Sigmoid()
B_ub_flat = torch.flatten(B)   # This B is initially defined as the default values, used now as upper bounds
B_mask = (B != 0.0).float()
B_mask_flat = torch.flatten(B_mask)

def B_restore(B):
    return relu(B)*B_mask  #TODO: check this for nzones = 1

def B_sigmoid(pred):
    return 2*B_ub_flat*sigmoid(pred)  #TODO: this zeroes out the null connections    TODO: this scaling is totally arbitrary
    #return 5*torch.ones(B_ub_flat.shape)*sigmoid(pred)

input_size  = nx + (nsteps+1)*(ny+ny+nd)  # ymax, ymin and d as input for each step, along with x0
B_predictor = Nets.ReLUnet(input_size, nB, hidden_sizes = [(input_size),2*(input_size),2*(input_size), (input_size), nB], batch_norm = True, initialize = True)

print("input_size")
print( input_size )
print("nB")
print( nB )

Q_weight = 50.0
R_weight =  1.0
diff_solver = get_building_MPC_layer(nsteps,nu,nx,ny,nd, umin,umax, A,C,E,F,G,y_ss, Q_weight,R_weight)

print("input size = {}".format(input_size))
print("output size = {}".format(nB))
"""
Create the upper-level training dataset
"""
B_optimizer = torch.optim.Adam(B_predictor.parameters(), lr=args.lr)
B_train_loader = DataLoader( list(zip(train_ymin, train_ymax, train_dist, train_x0)), shuffle=False, batch_size=batch_size_ul )
B_test_loader  = DataLoader( list(zip( test_ymin,  test_ymax,  test_dist,  test_x0)), shuffle=False, batch_size=batch_size_ul )


"""
Combined prediction and optimization with correction
"""
alpha = args.alpha #0.00001
use_correction=True
def predict_correct_solve(ymin, ymax, d, x0,   n_corr_steps = 2):
    batsize = len(x0)

    ymin_flat = torch.flatten(ymin, start_dim = 1)
    ymax_flat = torch.flatten(ymax, start_dim = 1)
    d_flat    = torch.flatten(d, start_dim = 1)
    x0_flat   = torch.flatten(x0, start_dim = 1)

    pred_input = torch.cat( (ymin_flat, ymax_flat, d_flat, x0_flat), dim=1 )
    pred_output = B_predictor(pred_input)
    B_pred = B_sigmoid(pred_output)
    B_pred = B_pred.view(batsize,nx,nu)
    if use_correction:
        B, viols_ymin_corr_list, viols_ymax_corr_list = B_correction(B_pred, x0, ymin, ymax, d, diff_solver, B_restore, alpha, n_corr_steps)
    else:
        B, viols_ymin_corr_list, viols_ymax_corr_list = B_pred, [], []

    solver_out = diff_solver(B, x0.squeeze(1), d, ymin, ymax)
    return B, solver_out






"""
Upper-level training loop
"""
penalty_weight = args.penalty #1000.0

test_viol_ymin_norm_list = []
test_viol_ymax_norm_list = []
test_viol_ymin_mean_list = []
test_viol_ymax_mean_list = []
test_Bsum_list = []
test_loss_list = []
test_viol_ymin_norm_std_list = []
test_viol_ymax_norm_std_list = []
test_viol_ymin_mean_std_list = []
test_viol_ymax_mean_std_list = []
test_Bsum_std_list = []
test_loss_std_list = []

train_viol_ymin_norm_list = []
train_viol_ymax_norm_list = []
train_viol_ymin_mean_list = []
train_viol_ymax_mean_list = []
train_Bsum_list = []
train_loss_list = []

epochs = args.epochs
for epoch in range(epochs):

    #if True:
        #(ymin, ymax, d, x0) = (test_ymin, test_ymax, test_dist, test_x0)
    batch_viol_ymin_norm_list = []
    batch_viol_ymax_norm_list = []
    batch_viol_ymin_mean_list = []
    batch_viol_ymax_mean_list = []
    batch_Bsum_list = []
    batch_loss_list = []
    for (ymin, ymax, d, x0) in B_test_loader:
        B, solver_out = predict_correct_solve(ymin, ymax, d, x0, n_corr_steps = 5)
        slack_lower = solver_out[3]
        slack_upper = solver_out[4]

        viols_ymin = torch.flatten(slack_lower, start_dim=1)
        viols_ymax = torch.flatten(slack_upper, start_dim=1)

        viol_ymin_sumsq = (viols_ymin**2).sum(1)
        viol_ymax_sumsq = (viols_ymax**2).sum(1)

        viol_ymin_norm = torch.norm(viols_ymin, dim=1, p=2)
        viol_ymax_norm = torch.norm(viols_ymax, dim=1, p=2)

        viol_ymin_mean = (viols_ymin).mean(1)
        viol_ymax_mean = (viols_ymax).mean(1)

        Bsum = torch.flatten(B, start_dim=1).sum(1)

        loss = penalty_weight*(viol_ymin_sumsq + viol_ymax_sumsq) + Bsum

        batch_viol_ymin_norm_list.append(viol_ymin_norm)
        batch_viol_ymax_norm_list.append(viol_ymax_norm)
        batch_viol_ymin_mean_list.append(viol_ymin_mean)
        batch_viol_ymax_mean_list.append(viol_ymax_mean)
        batch_Bsum_list.append(Bsum)
        batch_loss_list.append(loss)


    test_viol_ymin_norm_avg = torch.cat( tuple(batch_viol_ymin_norm_list), dim=0 ).mean().item()
    test_viol_ymax_norm_avg = torch.cat( tuple(batch_viol_ymax_norm_list), dim=0 ).mean().item()
    test_viol_ymin_mean_avg = torch.cat( tuple(batch_viol_ymin_mean_list), dim=0 ).mean().item()
    test_viol_ymax_mean_avg = torch.cat( tuple(batch_viol_ymax_mean_list), dim=0 ).mean().item()
    test_Bsum_avg           = torch.cat( tuple(batch_Bsum_list), dim=0 ).mean().item()
    test_loss_avg           = torch.cat( tuple(batch_loss_list), dim=0 ).mean().item()

    test_viol_ymin_norm_std = torch.cat( tuple(batch_viol_ymin_norm_list), dim=0 ).std().item()
    test_viol_ymax_norm_std = torch.cat( tuple(batch_viol_ymax_norm_list), dim=0 ).std().item()
    test_viol_ymin_mean_std = torch.cat( tuple(batch_viol_ymin_mean_list), dim=0 ).std().item()
    test_viol_ymax_mean_std = torch.cat( tuple(batch_viol_ymax_mean_list), dim=0 ).std().item()
    test_Bsum_std           = torch.cat( tuple(batch_Bsum_list), dim=0 ).std().item()
    test_loss_std           = torch.cat( tuple(batch_loss_list), dim=0 ).std().item()


    test_viol_ymin_norm_list.append(test_viol_ymin_norm_avg)
    test_viol_ymax_norm_list.append(test_viol_ymax_norm_avg)
    test_viol_ymin_mean_list.append(test_viol_ymin_mean_avg)
    test_viol_ymax_mean_list.append(test_viol_ymax_mean_avg)
    test_Bsum_list.append(test_Bsum_avg)
    test_loss_list.append(test_loss_avg)

    test_viol_ymin_norm_std_list.append(test_viol_ymin_norm_std)
    test_viol_ymax_norm_std_list.append(test_viol_ymax_norm_std)
    test_viol_ymin_mean_std_list.append(test_viol_ymin_mean_std)
    test_viol_ymax_mean_std_list.append(test_viol_ymax_mean_std)
    test_Bsum_std_list.append(test_Bsum_std)
    test_loss_std_list.append(test_loss_std)

    print("\n")
    print("Outer Epoch {}".format(epoch))
    print("ymin violations: \t norm = {} \t mean = {}".format(test_viol_ymin_norm_avg, test_viol_ymin_mean_avg))
    print("ymax violations: \t norm = {} \t mean = {}".format(test_viol_ymax_norm_avg, test_viol_ymax_mean_avg))
    print("Mean sum of B = {}".format(test_Bsum_avg))


    if epoch == epochs - 1: break


    for (ymin, ymax, d, x0) in B_train_loader:
        B, solver_out = predict_correct_solve(ymin, ymax, d, x0)
        slack_lower = solver_out[3]
        slack_upper = solver_out[4]

        viols_ymin = torch.flatten(slack_lower, start_dim=1)
        viols_ymax = torch.flatten(slack_upper, start_dim=1)

        viol_ymin_sumsq = (viols_ymin**2).sum(1).mean()
        viol_ymax_sumsq = (viols_ymax**2).sum(1).mean()

        viol_ymin_norm = torch.norm(viols_ymin, dim=1, p=2).mean()
        viol_ymax_norm = torch.norm(viols_ymax, dim=1, p=2).mean()

        viol_ymin_mean = (viols_ymin).mean(1).mean()
        viol_ymax_mean = (viols_ymax).mean(1).mean()

        Bsum = torch.flatten(B, start_dim=1).sum(1).mean()

        loss = penalty_weight*(viol_ymin_sumsq + viol_ymax_sumsq) + Bsum

        loss.backward()
        B_optimizer.step()
        B_optimizer.zero_grad()

        train_viol_ymin_norm_list.append(viol_ymin_norm.item())
        train_viol_ymax_norm_list.append(viol_ymax_norm.item())
        train_viol_ymin_mean_list.append(viol_ymin_mean.item())
        train_viol_ymax_mean_list.append(viol_ymax_mean.item())
        train_Bsum_list.append(Bsum.item())
        train_loss_list.append(loss.item())





loss_png_name = "loss_curves"
for (k,v) in vars(args).items():
    loss_png_name += "__" + k + "-" + str(v)
loss_png_name += ".png"






#loss_png_name = "loss_curves_{}.png".format(args.index)
fig, axs = plt.subplots(3,1,figsize=(9,12))
axs[0].tick_params(axis='y', labelsize=12)
axs[1].tick_params(axis='y', labelsize=12)
axs[2].tick_params(axis='y', labelsize=12)

test_viol_ymin_mean_up   = (np.array(test_viol_ymin_mean_list) + np.array(test_viol_ymin_mean_std_list)).tolist()
test_viol_ymin_mean_dwn  = (np.array(test_viol_ymin_mean_list) - np.array(test_viol_ymin_mean_std_list)).tolist()
axs[0].semilogy(range(len(test_viol_ymin_mean_list)), test_viol_ymin_mean_list, 'k-', label=r"$\frac{1}{N} \Sigma \; viol(ymin)$")
axs[0].fill_between(range(len(test_viol_ymin_mean_list)), test_viol_ymin_mean_dwn, test_viol_ymin_mean_up, color = 'lightslategrey')
test_viol_ymax_mean_up   = (np.array(test_viol_ymax_mean_list) + np.array(test_viol_ymax_mean_std_list)).tolist()
test_viol_ymax_mean_dwn  = (np.array(test_viol_ymax_mean_list) - np.array(test_viol_ymax_mean_std_list)).tolist()
axs[0].semilogy(range(len(test_viol_ymax_mean_list)), test_viol_ymax_mean_list, 'r-', label=r"$\frac{1}{N} \Sigma \; viol(ymax)$")
axs[0].fill_between(range(len(test_viol_ymax_mean_list)), test_viol_ymax_mean_dwn, test_viol_ymax_mean_up, color = 'pink')
test_viol_ymin_norm_up   = (np.array(test_viol_ymin_norm_list) + np.array(test_viol_ymin_norm_std_list)).tolist()
test_viol_ymin_norm_dwn  = (np.array(test_viol_ymin_norm_list) - np.array(test_viol_ymin_norm_std_list)).tolist()
axs[0].semilogy(range(len(test_viol_ymin_norm_list)), test_viol_ymin_norm_list, 'b-', label=r"$\|viol(ymin)\|_2$")
axs[0].fill_between(range(len(test_viol_ymin_norm_list)), test_viol_ymin_norm_dwn, test_viol_ymin_norm_up, color = 'lightskyblue')
test_viol_ymax_norm_up   = (np.array(test_viol_ymax_norm_list) + np.array(test_viol_ymax_norm_std_list)).tolist()
test_viol_ymax_norm_dwn  = (np.array(test_viol_ymax_norm_list) - np.array(test_viol_ymax_norm_std_list)).tolist()
axs[0].semilogy(range(len(test_viol_ymax_norm_list)), test_viol_ymax_norm_list, 'k-', label=r"$\|viol(ymax)\|_2$")
axs[0].fill_between(range(len(test_viol_ymax_norm_list)), test_viol_ymax_norm_dwn, test_viol_ymax_norm_up, color = 'lightslategrey')
axs[0].set_ylabel('Coupling Constraint Violations: Test Set')
axs[0].legend(fontsize=14)

test_Bsum_up   = (np.array(test_Bsum_list) + np.array(test_Bsum_std_list)).tolist()
test_Bsum_dwn  = (np.array(test_Bsum_list) - np.array(test_Bsum_std_list)).tolist()
axs[1].plot(range(len(test_Bsum_list)), test_Bsum_list, 'b-', label=r"$\Sigma \; B$")
axs[1].fill_between(range(len(test_Bsum_list)), test_Bsum_dwn, test_Bsum_up, color = 'lightskyblue')
axs[1].set_ylabel('Upper-Level Objective: Test Set')
axs[1].legend(fontsize=14)

axs[2].semilogy(range(len(test_loss_list)), test_loss_list, label = "Loss")
axs[2].set_ylabel('Loss Function: Test Set')
axs[2].set_xlabel('Training Epoch')
axs[2].legend(fontsize=14)
plt.savefig("./plt/test_"+loss_png_name)


fig, axs = plt.subplots(3,1,figsize=(9,12))
axs[0].tick_params(axis='y', labelsize=12)
axs[1].tick_params(axis='y', labelsize=12)
axs[2].tick_params(axis='y', labelsize=12)

axs[0].semilogy(range(len(train_viol_ymin_norm_list)), train_viol_ymin_mean_list, label = r"$\frac{1}{N} \Sigma \; viol(ymin)$")
axs[0].semilogy(range(len(train_viol_ymax_norm_list)), train_viol_ymax_mean_list, label = r"$\frac{1}{N} \Sigma \; viol(ymax)$")
axs[0].semilogy(range(len(train_viol_ymin_norm_list)), train_viol_ymin_norm_list, label = r"$\|viol(ymin)\|_2$")
axs[0].semilogy(range(len(train_viol_ymax_norm_list)), train_viol_ymax_norm_list, label = r"$\|viol(ymax)\|_2$")
axs[0].set_ylabel('Train Set Violations')
axs[0].legend(fontsize=14)

axs[1].plot(range(len(train_Bsum_list)), train_Bsum_list, label = r"$\Sigma \; B$")
axs[1].set_ylabel('Train Set Objective')
axs[1].legend(fontsize=14)

axs[2].semilogy(range(len(train_loss_list)), train_loss_list, label = "Loss")
axs[2].set_ylabel('Train Set Loss Function')
axs[2].set_xlabel('Upper-level Training Iteration')
axs[2].legend(fontsize=14)
plt.savefig("./plt/train_"+loss_png_name)




output_dict_name = "./pickle/outdict"
for (k,v) in vars(args).items():
    output_dict_name += "__" + k + "-" + str(v)
output_dict_name += ".p"


#output_dict_name = "pickle/outdict_{}.p".format(args.index)
output_dict = vars(args)
output_dict["test_viol_ymin_mean_list"] = test_viol_ymin_mean_list
output_dict["test_viol_ymax_mean_list"] = test_viol_ymax_mean_list
output_dict["test_viol_ymin_norm_list"] = test_viol_ymin_norm_list
output_dict["test_viol_ymax_norm_list"] = test_viol_ymax_norm_list
output_dict["test_Bsum_list"] = test_Bsum_list
output_dict["test_loss_list"] = test_loss_list
output_dict["test_viol_ymin_mean_std_list"] = test_viol_ymin_mean_std_list
output_dict["test_viol_ymax_mean_std_list"] = test_viol_ymax_mean_std_list
output_dict["test_viol_ymin_norm_std_list"] = test_viol_ymin_norm_std_list
output_dict["test_viol_ymax_norm_std_list"] = test_viol_ymax_norm_std_list
output_dict["test_Bsum_std_list"] = test_Bsum_std_list
output_dict["test_loss_std_list"] = test_loss_std_list


output_dict["train_viol_ymin_mean_list"] = train_viol_ymin_mean_list
output_dict["train_viol_ymax_mean_list"] = train_viol_ymax_mean_list
output_dict["train_viol_ymin_norm_list"] = train_viol_ymin_norm_list
output_dict["train_viol_ymax_norm_list"] = train_viol_ymax_norm_list
output_dict["train_Bsum_list"] = train_Bsum_list
output_dict["train_loss_list"] = train_loss_list
pickle.dump(output_dict,open(output_dict_name,'wb'))

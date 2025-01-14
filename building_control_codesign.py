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



torch.manual_seed(0)
np.random.seed(0)

# ground truth system model
sys = psl.systems['LinearSimpleSingleZone']()

# problem dimensions
nx = sys.nx                # number of states
nu = sys.nu                # number of control inputs
nd = sys.nD                # number of disturbances
ny = sys.ny                # number of controlled outputs
nref = ny                  # number of references
partial_observe = False
if partial_observe:        # Toggle partial observability of the disturbance
    d_idx = sys.d_idx
else:
    d_idx = range(nd)
nd_obs = len(d_idx)   # number of observable disturbances
nB = nu*nx  # This is the size of B as passed in to the policy network, should be the number of learnable elements

# extract exact state space model matrices:
A = torch.tensor(sys.A)
B = torch.tensor(sys.Beta)
C = torch.tensor(sys.C)
E = torch.tensor(sys.E)
F = torch.zeros(ny)
G = torch.zeros(nx)
y_ss = torch.zeros(ny)

# control action bounds
umin = torch.tensor(sys.umin)
umax = torch.tensor(sys.umax)

umax = umax / 5000    # JK rescaling reduces computation time dramatically
B = B * 5000          #   however, I haven't been able to verify equivalence, due to either solver instability when R!=0 or nonunique solutions when R=0


nsteps = 50
ntrain = 10000
ntest  = 1000#200
batch_size_ul = 100




"""
Generate the main data defining each control task:
bounds ymin, ymax;
disturbances dist;
initial state x0
"""
batched_ymin = torch.stack([torch.tensor(psl.signals.beta_walk_max_step(nsteps + 1, 1, min=18., max=22., max_step = 3.0, p = 0.1)) for _ in range(ntrain+ntest)])
batched_ymax = batched_ymin + 2.
batched_dist = torch.stack([torch.tensor(sys.get_D(nsteps+1)) for _ in range(ntrain+ntest)])*0.0   # JK: mult by 0 to turn off disturbances;   Notes: previously was nsteps (not nsteps+1) in DPC code!
batched_x0   = torch.stack([torch.tensor(sys.get_x0()).unsqueeze(0) for _ in range(ntrain+ntest)])
batched_x0[:,:,-1] = (batched_ymax[:,0,:] + batched_ymin[:,0,:]) / 2


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

def B_restore(B):
    return relu(B)

def B_sigmoid(pred):
    return 2*B_ub_flat*sigmoid(pred)
    #return 5*torch.ones(B_ub_flat.shape)*sigmoid(pred)

input_size  = nx + (nsteps+1)*(ny+ny+nd)  # ymax, ymin and d as input for each step, along with x0
B_predictor = Nets.ReLUnet(input_size, nB, hidden_sizes = [(input_size),2*(input_size),2*(input_size), (input_size), nB], batch_norm = True, initialize = True)

Q_weight = 50.0
R_weight =  1.0
diff_solver = get_building_MPC_layer(nsteps,nu,nx,ny,nd, umin,umax, A,C,E,F,G,y_ss, Q_weight,R_weight)

print("input size = {}".format(input_size))
print("output size = {}".format(nB))
"""
Create the upper-level training dataset
"""
B_optimizer = torch.optim.Adam(B_predictor.parameters(), lr=1e-3)
B_train_loader = DataLoader( list(zip(train_ymin, train_ymax, train_dist, train_x0)), shuffle=False, batch_size=batch_size_ul )


"""
Combined prediction and optimization with correction
"""
alpha = 0.00001
n_corr_steps = 2
use_correction=True
def predict_correct_solve(ymin, ymax, d, x0):
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
penalty_weight = 1000.0
test_viol_ymin_norm_list = []
test_viol_ymax_norm_list = []
test_viol_ymin_mean_list = []
test_viol_ymax_mean_list = []
test_Bsum_list = []
test_loss_list = []

train_viol_ymin_norm_list = []
train_viol_ymax_norm_list = []
train_viol_ymin_mean_list = []
train_viol_ymax_mean_list = []
train_Bsum_list = []
train_loss_list = []
epochs = 5 #10
for epoch in range(epochs):

    if True:
        (ymin, ymax, d, x0) = (test_ymin, test_ymax, test_dist, test_x0)
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

        test_viol_ymin_norm_list.append(viol_ymin_norm.item())
        test_viol_ymax_norm_list.append(viol_ymax_norm.item())
        test_viol_ymin_mean_list.append(viol_ymin_mean.item())
        test_viol_ymax_mean_list.append(viol_ymax_mean.item())
        test_Bsum_list.append(Bsum.item())
        test_loss_list.append(loss.item())

        print("\n")
        print("Outer Epoch {}".format(epoch))
        print("ymin violations: \t norm = {} \t mean = {}".format(viol_ymin_norm.item(), viol_ymin_mean.item()))
        print("ymax violations: \t norm = {} \t mean = {}".format(viol_ymax_norm.item(), viol_ymax_mean.item()))
        print("Mean sum of B = {}".format(Bsum.item()))




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







fig, axs = plt.subplots(3,1,figsize=(9,12))
axs[0].tick_params(axis='y', labelsize=12)
axs[1].tick_params(axis='y', labelsize=12)
axs[2].tick_params(axis='y', labelsize=12)

axs[0].semilogy(range(len(test_viol_ymin_norm_list)), test_viol_ymin_mean_list, label = r"$\frac{1}{N} \Sigma \; viol(ymin)$")
axs[0].semilogy(range(len(test_viol_ymax_norm_list)), test_viol_ymax_mean_list, label = r"$\frac{1}{N} \Sigma \; viol(ymax)$")
axs[0].semilogy(range(len(test_viol_ymin_norm_list)), test_viol_ymin_norm_list, label = r"$\|viol(ymin)\|_2$")
axs[0].semilogy(range(len(test_viol_ymax_norm_list)), test_viol_ymax_norm_list, label = r"$\|viol(ymax)\|_2$")
axs[0].set_ylabel('Test Set Violations')
axs[0].legend(fontsize=14)

axs[1].plot(range(len(test_Bsum_list)), test_Bsum_list, label = r"$\Sigma \; B$")
axs[1].set_ylabel('Test Set Objective')
axs[1].legend(fontsize=14)

axs[2].semilogy(range(len(test_loss_list)), test_loss_list, label = "Loss")
axs[2].set_ylabel('Test Set Loss Function')
axs[2].set_xlabel('Upper-level Training Epoch')
axs[2].legend(fontsize=14)
plt.show()





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
plt.show()

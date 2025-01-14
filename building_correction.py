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

relu = torch.nn.ReLU()

def B_restore(B):
    return relu(B)


def B_correction(B, x0, ymin, ymax, d, diff_solver, B_restore, alpha, n_corr_steps):

        y0 = x0[:,:,[-1]]

        viols_ymin_list = []
        viols_ymax_list = []
        for _ in range(n_corr_steps):

            B = B_restore(B)

            solver_out = diff_solver(B, x0.squeeze(1), d, ymin, ymax)
            u = solver_out[0]
            x = solver_out[1]
            y = solver_out[2]
            slack_lower = solver_out[3]
            slack_upper = solver_out[4]
            viols_ymin = torch.flatten(slack_lower, start_dim=1)
            viols_ymax = torch.flatten(slack_upper, start_dim=1)
            #viols_ymin = relu(ymin - y)
            #viols_ymax = relu(y - ymax)
            viols = torch.cat( (viols_ymin, viols_ymax), dim = 1)  #  viols_ymin

            grad_viol = 2*(viols)  # d/dB gradient of \| viols_ymin \|**2 + \| viols_ymax\|**2

            viols_ymin_list.append( viols_ymin.mean().item() )
            viols_ymax_list.append( viols_ymax.mean().item() )

            grads_B = torch.autograd.grad( viols, B, grad_viol, retain_graph = True )[0]
            B = B - alpha*grads_B

        B = B_restore(B)

        return B, viols_ymin_list, viols_ymax_list


# Sketch, needs testing
def compute_traj(B, x0, ymin, ymax, d, cl_system):
    batsize = len(x0)

    y0 = x0[:, :, [-1]]
    data = {'x': x0,
            'y': y0,
            'ymin': ymin,
            'ymax': ymax,
            'd': d,
            'B':B.unsqueeze(1).repeat(1,nsteps+1,1)}

    trajectories = cl_system(data)
    x = trajectories['x'].reshape(batsize, nsteps + 1, nx)
    y = trajectories['x'].reshape(batsize, nsteps + 1, ny)
    u = trajectories['x'].reshape(batsize, nsteps + 1, nu)

    return x,y,u





if __name__ == "__main__":

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

    umax = umax / 1000    # JK rescaling reduces computation time dramatically
    B = B * 1000          #   however, I haven't been able to verify equivalence, due to either solver instability when R!=0 or nonunique solutions when R=0


    nsteps = 50
    n_samples = 10


    Q_weight = 50.0
    R_weight =  1.0
    diff_solver = get_building_MPC_layer(nsteps,nu,nx,ny,nd, umin,umax, A,C,E,F,G,y_ss, Q_weight,R_weight)


    batched_ymin = torch.stack([torch.tensor(psl.signals.beta_walk_max_step(nsteps + 1, 1, min=18., max=22., max_step = 3.0, p = 0.1)) for _ in range(n_samples)])
    batched_ymax = batched_ymin + 2.
    batched_dist = torch.stack([torch.tensor(sys.get_D(nsteps+1)) for _ in range(n_samples)])*0.0   # JK: mult by 0 to turn off disturbances
    batched_x0   = torch.stack([torch.tensor(sys.get_x0()).unsqueeze(0) for _ in range(n_samples)])


    x0   = batched_x0
    y0    = batched_x0[:,:,[-1]]
    ymin = batched_ymin
    ymax = batched_ymax
    d    = batched_dist

    B    = torch.rand(B.shape)*B.repeat(n_samples, 1, 1)
    Bsave=B

    solver_out = diff_solver(B, x0.squeeze(1), d, ymin, ymax)

    alpha = 0.000001
    n_corr_steps = 20
    B.requires_grad = True
    B_corr, viols_ymin_list, viols_ymax_list = B_correction(B, x0, ymin, ymax, d, diff_solver, B_restore, alpha, n_corr_steps)
    plt.semilogy(range(len(viols_ymin_list)), viols_ymin_list, label = "ymin violations")
    plt.semilogy(range(len(viols_ymax_list)), viols_ymax_list, label = "ymax violations")
    plt.legend()
    plt.show()

    print("(Bsave - B).abs().max()")
    print( (Bsave - B).abs().max() )

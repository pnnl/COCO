"""
Neural Ordinary Differentiable predictive control (NO-DPC)

Reference tracking of nonlinear ODE system with explicit neural control policy via DPC algorithm

system: Two Tank model
example inspired by: https://apmonitor.com/do/index.php/Main/LevelControl
"""

import torch
import torch.nn as nn
import numpy as np

import neuromancer.psl as psl
from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.modules.activations import activations
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuromancer.dynamics import ode, integrators
from neuromancer.plot import pltCL, pltPhase

import twotank_utils

import matplotlib.pyplot as plt

class TwoTankPredict(ode.ODESystem):
    def __init__(self, insize=6, outsize=2):
        """
        :param insize:
        :param outsize:
        """
        super().__init__(insize=insize, outsize=outsize)

    def ode_equations(self, x, u, c):

        c1 = c[:, [0]]
        c2 = c[:, [1]]

        # heights in tanks
        h1 = torch.clip(x[:, [0]], min=0, max=1.0)
        h2 = torch.clip(x[:, [1]], min=0, max=1.0)
        # Inputs (2): pump and valve
        pump = torch.clip(u[:, [0]], min=0, max=1.0)
        valve = torch.clip(u[:, [1]], min=0, max=1.0)
        # equations
        dhdt1 = c1 * (1.0 - valve) * pump - c2 * torch.sqrt(h1)
        dhdt2 = c1 * valve * pump + c2 * torch.sqrt(h1) - c2 * torch.sqrt(h2)
        return torch.cat([dhdt1, dhdt2], dim=-1)



def train_DPC(net, train_data, dev_data, epochs, nsteps = 50, patience=50):
    """
    # # #  Ground truth system model
    """
    gt_model = psl.nonautonomous.TwoTank()


    # sampling rate
    ts = gt_model.params[1]['ts']
    # problem dimensions
    nx = gt_model.nx    # number of states
    nu = gt_model.nu    # number of control inputs
    nref = nx           # number of references
    # constraints bounds
    umin = 0
    umax = 1.
    xmin = 0
    xmax = 1.

    """
    # # #  Dataset
    """
    # torch dataloaders
    batch_size = 200
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               collate_fn=train_data.collate_fn,
                                               shuffle=False)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=batch_size,
                                             collate_fn=dev_data.collate_fn,
                                             shuffle=False)

    """
    # # #  System model and Control policy in Neuromancer
    """
    # white-box ODE model with no-plant model mismatch
    two_tank_ode = TwoTankPredict()
    # integrate continuous time ODE
    integrator = integrators.RK4(two_tank_ode, h=torch.tensor(ts))  # using 4th order runge kutta integrator
    # symbolic system model
    model = Node(integrator, ['x', 'u', 'c'], ['x'], name='model')

    # neural net control policy
    """
    net = blocks.MLP_bounds(insize=nx + nref, outsize=nu, hsizes=[32, 32],
                        nonlin=activations['gelu'], min=umin, max=umax)
    """
    policy = Node(net, ['x', 'r', 'c'], ['u'], name='policy')

    # closed-loop system model
    cl_system = System([policy, model], nsteps=nsteps,
                       name='cl_system')
    #cl_system.show()

    """
    # # #  Differentiable Predictive Control objectives and constraints
    """
    # variables
    x   = variable('x')
    ref = variable("r")
    c   = variable("c")

    state_lower_bound_penalty = 10.*(x > xmin)
    state_upper_bound_penalty = 10.*(x < xmax)
    terminal_lower_bound_penalty = 10.*(x[:, [-1], :] > ref-0.01)
    terminal_upper_bound_penalty = 10.*(x[:, [-1], :] < ref+0.01)

    state_lower_bound_penalty.name = 'x_min'
    state_upper_bound_penalty.name = 'x_max'
    terminal_lower_bound_penalty.name = 'x_N_min'
    terminal_upper_bound_penalty.name = 'x_N_max'
    # list of constraints and objectives
    objectives = [] #[regulation_loss]
    constraints = [
        state_lower_bound_penalty,
        state_upper_bound_penalty,
        terminal_lower_bound_penalty,
        terminal_upper_bound_penalty,
    ]

    """
    # # #  Differentiable optimal control problem
    """
    # data (x_k, r_k) -> parameters (xi_k) -> policy (u_k) -> dynamics (x_k+1)
    nodes = [cl_system]
    # create constrained optimization loss
    loss = PenaltyLoss(objectives, constraints)
    # construct constrained optimization problem
    problem = Problem(nodes, loss)
    # plot computational graph
    #problem.show()

    """
    # # #  Solving the problem
    """
    optimizer = torch.optim.AdamW(problem.parameters(), lr=0.01)
    #  Neuromancer trainer
    callback = twotank_utils.CallbackChild()
    trainer = Trainer(
        problem,
        train_loader, dev_loader,
        optimizer=optimizer,
        callback=callback,
        epochs=epochs,
        train_metric='train_loss',
        eval_metric='dev_loss',
        warmup=3,
        patience=patience
    )
    # Train control policy
    best_model = trainer.train()
    # load best trained model
    trainer.model.load_state_dict(best_model)
    outputs = {}
    outputs['dev_losses_epoch'] = trainer.dev_losses_epoch

    return net, outputs


if __name__ == "__main__":

    """
    # # #  Ground truth system model
    """
    gt_model = psl.nonautonomous.TwoTank()
    # sampling rate
    ts = gt_model.params[1]['ts']
    # problem dimensions
    nx = gt_model.nx    # number of states
    nu = gt_model.nu    # number of control inputs
    nref = nx           # number of references

    c = torch.Tensor([0.08,0.04])

    nsteps = 50         # prediction horizon
    n_samples = 30000   # number of sampled scenarios

    #  sampled references for training the policy
    list_refs = [torch.rand(1, 1)*torch.ones(nsteps+1, nref) for k in range(n_samples)]
    ref = torch.cat(list_refs)
    batched_ref = ref.reshape([n_samples, nsteps+1, nref])

    # Training dataset
    train_data = {'x': torch.rand(n_samples, 1, nx),
                  'r': torch.rand(n_samples,2).unsqueeze(1).repeat(1,nsteps+1,1),#batched_ref,
                  'c': c*torch.ones(n_samples, nsteps+1, 2)}
    train_dataset = DictDataset(train_data, name='train')

    # references for dev set
    list_refs = [torch.rand(1, 1)*torch.ones(nsteps+1, nref) for k in range(n_samples)]
    ref = torch.cat(list_refs)
    batched_ref = ref.reshape([n_samples, nsteps+1, nref])

    # Development dataset
    dev_data = {'x': torch.rand(n_samples, 1, nx),
                'r': torch.rand(n_samples,2).unsqueeze(1).repeat(1,nsteps+1,1),#batched_ref,
                'c': c*torch.ones(n_samples, nsteps+1, 2)}
    dev_dataset = DictDataset(dev_data, name='dev')

    umin = 0
    umax = 1.
    xmin = 0
    xmax = 1.
    nc = nx
    net = blocks.MLP_bounds(insize=nx + nref + nc,
                            #insize=nx + nref,
                            outsize=nu, hsizes=[32, 32],
                            nonlin=activations['gelu'], min=umin, max=umax)

    epochs = 7
    net, outputs = train_DPC(net, train_dataset, dev_dataset, epochs, nsteps=nsteps)



    """
    Rebuild and evaluate the learned control model
    """

    two_tank_ode = TwoTankPredict()
    integrator = integrators.RK4(two_tank_ode, h=torch.tensor(ts))  # using 4th order runge kutta integrator
    model  = Node(integrator, ['x', 'u', 'c'], ['x'], name='model')
    policy = Node(net, ['x', 'r', 'c'], ['u'], name='policy')
    cl_system = System([policy, model], nsteps=nsteps,
                       name='cl_system')


    trajectories = cl_system(dev_data)
    x_dev   = trajectories['x'].reshape(n_samples, nsteps + 1, nx)
    target_x_dev = batched_ref


    for k in range(len(x_dev)):
        plt.plot(  target_x_dev[k][:,0].detach(), target_x_dev[k][:,1].detach(),  'r*-', label=r"Target" )
        plt.plot(    x_dev[k][:,0].detach(),   x_dev[k][:,1].detach(),  'b*-', label=r"Predicted trajectory" )
        # set axis limits
        plt.xlim(0,1.0)
        plt.ylim(0,1.0)
        plt.legend()
        plt.show()

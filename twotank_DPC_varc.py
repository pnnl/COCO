
"""
Variation from twotank_DPC.py
We evaluate the DPC learning over a distribution of plant parameters c, with static x0 and xf,
rather than with static c and variable x0/xf
"""



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
        #self.c1 = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        #self.c2 = nn.Parameter(torch.tensor([0.1]), requires_grad=True)

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


    #nsteps = 50  # prediction horizon


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
    #two_tank_ode = ode.TwoTankParam()
    #two_tank_ode.c1 = nn.Parameter(torch.tensor(gt_model.c1), requires_grad=False)
    #two_tank_ode.c2 = nn.Parameter(torch.tensor(gt_model.c2), requires_grad=False)

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
    # objectives
    regulation_loss = 5. * ((x == ref) ^ 2)  # target posistion
    # constraints
    state_lower_bound_penalty = 10.*(x > xmin)
    state_upper_bound_penalty = 10.*(x < xmax)
    terminal_lower_bound_penalty = 10.*(x[:, [-1], :] > ref-0.01)
    terminal_upper_bound_penalty = 10.*(x[:, [-1], :] < ref+0.01)
    # objectives and constraints names for nicer plot
    regulation_loss.name = 'ref_tracking'
    state_lower_bound_penalty.name = 'x_min'
    state_upper_bound_penalty.name = 'x_max'
    terminal_lower_bound_penalty.name = 'x_N_min'
    terminal_upper_bound_penalty.name = 'x_N_max'
    # list of constraints and objectives
    objectives = [regulation_loss]
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
    #optimizer = torch.optim.SGD(problem.parameters(), lr=0.01)
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

    """
    Test Closed Loop System
    """
    """
    print('\nTest Closed Loop System \n')
    nsteps = 750
    step_length = 150
    # generate reference
    np_refs = psl.signals.step(nsteps+1, 1, min=xmin, max=xmax, randsteps=5)
    R = torch.tensor(np_refs, dtype=torch.float32).reshape(1, nsteps+1, 1)
    torch_ref = torch.cat([R, R], dim=-1)
    c = torch.Tensor([0.0,0.04])
    # generate initial data for closed loop simulation
    data = {'x': torch.rand(1, 1, nx, dtype=torch.float32),
            'r': torch_ref,
            'c': c*torch.ones(1, nsteps+1, 2)}
    cl_system.nsteps = nsteps
    # perform closed-loop simulation
    trajectories = cl_system(data)

    # constraints bounds
    Umin = umin * np.ones([nsteps, nu])
    Umax = umax * np.ones([nsteps, nu])
    Xmin = xmin * np.ones([nsteps+1, nx])
    Xmax = xmax * np.ones([nsteps+1, nx])
    # plot closed loop trajectories
    pltCL(Y=trajectories['x'].detach().reshape(nsteps + 1, nx),
          R=trajectories['r'].detach().reshape(nsteps + 1, nref),
          U=trajectories['u'].detach().reshape(nsteps, nu),
          Umin=Umin, Umax=Umax, Ymin=Xmin, Ymax=Xmax)
    # plot phase portrait
    pltPhase(X=trajectories['x'].detach().reshape(nsteps + 1, nx))
    print("trajectories['x'].detach().reshape(nsteps + 1, nx)")
    print( trajectories['x'].detach().reshape(nsteps + 1, nx) )
    """

    outputs = {}
    outputs['dev_losses_epoch'] = trainer.dev_losses_epoch

    return net, outputs, trainer.model_list




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


    # default c1:  0.08
    # default c2:  0.04

    x0 = torch.Tensor([0.0,0.0])
    xf = torch.Tensor([0.0,0.5])

    nsteps = 50  # prediction horizon
    n_train = 10000    # number of sampled scenarios
    n_dev    = 1000

    cmax = 0.12    #0.12
    cmin = 0.04    #0.02

    c2 = cmin*torch.ones(n_train,1)  #(cmax-cmin)*torch.rand(n_train,1) + cmin    #
    c2step = 0.1*torch.rand(n_train,1)              #
    c1 = c2 + c2step                                 #
    c = torch.cat((c1,c2),dim=1)

    #c = (cmax-cmin)*torch.rand(n_train,2) + cmin
    c_train = c.unsqueeze(1).repeat(1, nsteps+1, 1)
    x0_train = x0*torch.ones(n_train,        1, 2)
    xr_train = xf*torch.ones(n_train, nsteps+1, 2)
    xr_train[:,(nsteps//2):,:] = 0.5


    c2 = cmin*torch.ones(n_dev,1)  #(cmax-cmin)*torch.rand(n_dev,1) + cmin    #
    c2step = 0.1*torch.rand(n_dev,1)              #
    c1 = c2 + c2step                                 #
    c = torch.cat((c1,c2),dim=1)

    #c = (cmax-cmin)*torch.rand(n_dev,2) + cmin
    c_dev = c.unsqueeze(1).repeat(1, nsteps+1, 1)
    x0_dev = x0*torch.ones(n_dev,        1, 2)
    xr_dev = xf*torch.ones(n_dev, nsteps+1, 2)
    xr_dev[:,(nsteps//2):,:] = 0.5

    # Training dataset
    train_data = {'x': x0_train,
                  'r': xr_train,
                  'c': c_train}  # Duplicate c across the 'timesteps' dimension
    train_dataset = DictDataset(train_data, name='train')

    # Development dataset
    dev_data = {'x': x0_dev,
                'r': xr_dev,
                'c':  c_dev}
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

    epochs = 5#10#20
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

    print("net")
    print( net )
    print("model")
    print( model )
    print("policy")
    print( policy )
    print("cl_system")
    print( cl_system )
    print("dev_data")
    print( dev_data )
    input("waiting")

    trajectories = cl_system(dev_data)
    x_dev   = trajectories['x'].reshape(n_dev, nsteps + 1, nx)
    target_x_dev = xr_dev

    print("learned trajectories")
    print( x_dev.mean(0).detach() )
    print("target trajectories")
    print( target_x_dev.mean(0).detach() )

    pltPhase(X=x_dev.mean(0).detach())
    pltPhase(X=target_x_dev.mean(0).detach())

    for k in range(len(x_dev)):
        print("target_x_dev[k]")
        print( target_x_dev[k] )
        print("x_dev[k]")
        print( x_dev[k] )
        print("c_dev[k]")
        print( c_dev[k] )
        plt.plot(  x_dev[k][:,0].detach(), x_dev[k][:,1].detach(),  'b*-' )

        #plt.plot(  x_dev[k][0].detach(),  x_dev[k][1].detach(), 'b*-' )
        #plt.plot(  target_x_dev[k][0].detach(),  target_x_dev[k][1].detach(), 'r*-' )
        plt.xlim(0,1)
        plt.ylim(0,1)
        #pltPhase(X=x_dev[k].detach())
        #pltPhase(X=target_x_dev[k].detach())
        plt.show()

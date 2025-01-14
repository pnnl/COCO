# Branched from v2 to try a supervised training based on solutions from new cvxpylayer

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

from building_MPC_layer import get_building_MPC_layer, get_building_MPC_layer_B
import matplotlib.pyplot as plt

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
print("A")
print( A )
print("B")
print( B )
print("C")
print( C )
print("E")
print( E )


# control action bounds
umin = torch.tensor(sys.umin)
umax = torch.tensor(sys.umax)

print("nx")
print( nx )
print("ny")
print( ny )
print("nu")
print( nu )
print("nd")
print( nd )
print("umin")
print( umin )
print("umax")
print( umax )
print("d_idx")
print( d_idx )
print("nd_obs")
print( nd_obs )

umax = umax / 1000    # JK rescaling reduces computation time dramatically
B = B * 1000          #   however, I haven't been able to verify equivalence, due to either solver instability when R!=0 or nonunique solutions when R=0



def get_data(sys, nsteps, n_samples, batch_size, name="train"):
    #  sampled references for training the policy
    #batched_ymin = xmin_range.sample((n_samples, 1, nref)).repeat(1, nsteps + 1, 1)
    batched_ymin = torch.stack([torch.tensor(psl.signals.beta_walk_max_step(nsteps + 1, 1, min=18., max=22., max_step = 3.0, p = 0.1)) for _ in range(n_samples)])
    batched_ymax = batched_ymin + 2.

    # sampled disturbance trajectories from the simulation model
    batched_dist = torch.stack([torch.tensor(sys.get_D(nsteps)) for _ in range(n_samples)])*0.0   # JK: mult by 0 to turn off disturbances

    # sampled initial conditions
    batched_x0 = torch.stack([torch.tensor(sys.get_x0()).unsqueeze(0) for _ in range(n_samples)])


    #data = {"x": batched_x0,
    #        "y": batched_x0[:,:,[-1]],
    #     "ymin": batched_ymin,
    #     "ymax": batched_ymax,
    #        "d": batched_dist},
    #       name=name,
    #)

    data = DictDataset(
        {"x": batched_x0,
         "y": batched_x0[:,:,[-1]],
         "ymin": batched_ymin,
         "ymax": batched_ymax,
         "d": batched_dist},
        name=name,
    )


    return data #DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn, shuffle=False)    #JK


nsteps = 100  # prediction horizon
n_samples = 100 #1000    # number of sampled scenarios   #TODO reset
batch_size = 100

# range for lower comfort bound
#xmin_range = torch.distributions.Uniform(18., 22.)







#train_loader, dev_loader = [
#    get_data(sys, nsteps, n_samples, xmin_range, batch_size, name=name)
#    for name in ("train", "dev")
#]
train_data, dev_data = [
    get_data(sys, nsteps, n_samples, batch_size, name=name)
    for name in ("train", "dev")
]

x0_train = train_data.datadict["x"]
y0_train = train_data.datadict["y"]
d_train = train_data.datadict["d"]
ymax_train = train_data.datadict["ymax"]
ymin_train = train_data.datadict["ymin"]

x0_dev = dev_data.datadict["x"]
y0_dev = dev_data.datadict["y"]
d_dev = dev_data.datadict["d"]
ymax_dev = dev_data.datadict["ymax"]
ymin_dev = dev_data.datadict["ymin"]

# TODO: look into key error in training - data has no key "B"?  Try building DictDataset just once without the above appends


# TODO: get more realistic xmin_range for training and rescale umax, B

# Get precomputed solutions for supervised training
Q_weight = 50.0
R_weight =  1.0
cvxlayer = get_building_MPC_layer_B(nsteps,nu,nx,ny,nd, umin,umax, A,C,E,F,G,y_ss, Q_weight,R_weight)

B_train = torch.rand(B.shape)*B.repeat(n_samples, 1, 1)     # batch of B matrices
B_dev   = torch.rand(B.shape)*B.repeat(n_samples, 1, 1)
#B_expand_train = B_train.unsqueeze(1).repeat(1,5,1,1)
#B_expand_dev   =   B_dev.unsqueeze(1).repeat(1,5,1,1)



# We treat B as vectors in the neuomancer dataset to simplify coding.
B_flat_train = torch.flatten(B_train, start_dim=1)
B_flat_dev   = torch.flatten(B_dev,   start_dim=1)


# Expand them into repeats over nsteps for input to neuromancer system.
B_flat_expand_train = B_flat_train.unsqueeze(1).repeat(1,nsteps+1,1)
B_flat_expand_dev   = B_flat_dev.unsqueeze(1).repeat(1,nsteps+1,1)




train_sols  = cvxlayer(B_train, x0_train.squeeze(1), y0_train.squeeze(1), d_train, ymin_train, ymax_train)   # train_sols = cvxlayer_pre(x_init,y_init,d,ymin,ymax)
train_u_opt = train_sols[0]
train_x_opt = train_sols[1]
train_data.datadict["x_opt"] = train_x_opt
train_data.datadict["u_opt"] = train_u_opt
train_data.datadict["B"]     = B_flat_expand_train

dev_sols  = cvxlayer(B_dev, x0_dev.squeeze(1), y0_dev.squeeze(1), d_dev, ymin_dev, ymax_dev)   # train_sols = cvxlayer_pre(x_init,y_init,d,ymin,ymax)
dev_u_opt = dev_sols[0]
dev_x_opt = dev_sols[1]
dev_data.datadict["x_opt"] = dev_x_opt
dev_data.datadict["u_opt"] = dev_u_opt
dev_data.datadict["B"]     = B_flat_expand_dev

#print("dev_u_opt")
#print( dev_u_opt )
#print("dev_x_opt")
#print( dev_x_opt )
#input("waiting")




print("nB")
print( nB )
print("ny + 2*nref + nd_obs")
print( ny + 2*nref + nd_obs )
print("ny + 2*nref + nd_obs + nB")
print( ny + 2*nref + nd_obs + nB )
print("x0_dev")
print( x0_dev )
print("d_dev")
print( d_dev )
print("B_dev")
print( B_dev )
print("B_flat_dev")
print( B_flat_dev )
print("d_dev.shape")
print( d_dev.shape )
print("B_dev.shape")
print( B_dev.shape )
print("B_flat_dev.shape")
print( B_flat_dev.shape )




print("train_data.datadict")
print( train_data.datadict )
print("waiting")

# Next: look at the solutions from cvxlayer to check if they're within bounds where possible
#       randomize B and test the solver stability
#       include B as a training input and try to learn solutions as a function of both B, and the y-bounds


train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=train_data.collate_fn, shuffle=False)
dev_loader   = DataLoader(dev_data,   batch_size=batch_size, collate_fn=dev_data.collate_fn,   shuffle=False)






# state-space model of the building dynamics:
#   x_k+1 =  A x_k + B u_k + E d_k
xnext = lambda x, u, d, B: x @ A.T + (torch.bmm( u.unsqueeze(1), B.reshape(len(u),nx,nu).permute(0,2,1) )).squeeze(1) + d @ E.T
#xnext = lambda x, u, d, B: x @ A.T + u @ B.T + d @ E.T
state_model = Node(xnext, ['x', 'u', 'd', 'B'], ['x'], name='SSM')   # JK Shouldn't it be d_obs rather than d as input?
                                                                # nvm, policy takes d_obs but dynamics take d

#   y_k = C x_k
ynext = lambda x: x @ C.T
output_model = Node(ynext, ['x'], ['y'], name='y=Cx')

# partially observable disturbance model
dist_model = lambda d: d[:, d_idx]
dist_obs = Node(dist_model, ['d'], ['d_obs'], name='dist_obs')

print("next(train_loader)")
print( next(train_loader) )
print("dist_obs.input_keys")
print( dist_obs.input_keys )
input("waiting")

# partially observable disturbance model
#B2mat = lambda B:  B.view(len(B),nx,nu)
#shape_B = Node(B2mat, ['B'], ['B_mat'], name='shape_B')

# neural net control policy
net = blocks.MLP_bounds(
    insize=ny + 2*nref + nd_obs + nB,   # JK
    #insize=ny + 2*nref + nd_obs,
    outsize=nu,
    hsizes=[64, 64],
    nonlin=nn.GELU,
    min=umin,
    max=umax,
)
policy = Node(net, ['y', 'ymin', 'ymax', 'd_obs', 'B'], ['u'], name='policy')
#policy = Node(net, ['y', 'ymin', 'ymax', 'd_obs'], ['u'], name='policy')


# closed-loop system model
#cl_system = System([dist_obs, shape_B, policy, state_model, output_model],
#                    nsteps=nsteps,
#                    name='cl_system')
cl_system = System([dist_obs, policy, state_model, output_model],
                    nsteps=nsteps,
                    name='cl_system')
#cl_system.show()



# variables
x = variable('x')
y = variable('y')
u = variable('u')
ymin = variable('ymin')
ymax = variable('ymax')

x_opt = variable("x_opt")
u_opt = variable("u_opt")

Q_u = 0.01
Q_du = 0.1
Q_ymin = 50.0
Q_ymax = 50.0

# objectives (unsupervised version)
action_loss = 0.01 * (u == 0.0)  # energy minimization
du_loss = 0.1 * (u[:,:-1,:] - u[:,1:,:] == 0.0)  # delta u minimization to prevent agressive changes in control actions

# objectives (supervised version)
x_supervision_loss = 1.0 * (x == x_opt)
u_supervision_loss = 1.0 * (u == u_opt)

# thermal comfort constraints
state_lower_bound_penalty = 50.*(y > ymin)
state_upper_bound_penalty = 50.*(y < ymax)

# objectives and constraints names for nicer plot
action_loss.name = 'action_loss'
du_loss.name = 'du_loss'
state_lower_bound_penalty.name = 'x_min'
state_upper_bound_penalty.name = 'x_max'

# list of constraints and objectives
#objectives = [action_loss, du_loss]  # unsupervised
objectives  = [x_supervision_loss, u_supervision_loss]  # supervised
constraints = [state_lower_bound_penalty, state_upper_bound_penalty]



# data (x_k, r_k) -> parameters (xi_k) -> policy (u_k) -> dynamics (x_k+1)
nodes = [cl_system]
# create constrained optimization loss
loss = PenaltyLoss(objectives, constraints)
# construct constrained optimization problem
problem = Problem(nodes, loss)
# plot computational graph
#problem.show()




optimizer = torch.optim.AdamW(problem.parameters(), lr=0.001)
#  Neuromancer trainer
trainer = Trainer(
    problem,
    train_loader,
    dev_loader,
    optimizer=optimizer,
    epochs=1,#200, # TODO
    train_metric='train_loss',
    eval_metric='dev_loss',
    warmup=200,
)



# Train control policy
best_model = trainer.train()
# load best trained model
trainer.model.load_state_dict(best_model)





nsteps_test = 50 #2000

# generate reference
np_refs = psl.signals.step(nsteps_test+1, 1, min=18., max=22., randsteps=5)
#np_refs = xmin_range.sample((nsteps_test+1, 1))
ymin_test = torch.tensor(np_refs, dtype=torch.float32).reshape(1, nsteps_test+1, 1)
ymax_test = ymin_test+5.0#+2.0   # TODO
# generate disturbance signal
d_test = torch.tensor(sys.get_D(nsteps_test+1)).unsqueeze(0)
# initial data for closed loop simulation
x0_test = torch.tensor(sys.get_x0()).reshape(1, 1, nx)
data = {'x': x0_test,
        'y': x0_test[:, :, [-1]],
        'ymin': ymin_test,
        'ymax': ymax_test,
        'd': d_test}
cl_system.nsteps = nsteps_test
# perform closed-loop simulation
trajectories = cl_system(data)




def plot_solution(ymin_traj,ymax_traj, x_traj,y_traj,d_traj,u_traj):

    # constraints bounds
    Umin = umin * np.ones([nsteps_test, nu])
    Umax = umax * np.ones([nsteps_test, nu])
    Ymin = ymin_traj #trajectories['ymin'].detach().reshape(nsteps_test+1, nref)
    Ymax = ymax_traj #trajectories['ymax'].detach().reshape(nsteps_test+1, nref)
    # plot closed loop trajectories
    fig, ax = pltCL(Y= y_traj, #trajectories['y'].detach().reshape(nsteps_test+1, ny),
            R=Ymax,
            X= x_traj, #trajectories['x'].detach().reshape(nsteps_test+1, nx),
            D= d_traj, #trajectories['d'].detach().reshape(nsteps_test+1, nd),
            U= u_traj, #trajectories['u'].detach().reshape(nsteps_test, nu),
            Umin=Umin, Umax=Umax, Ymin=Ymin, Ymax=Ymax)
    # add custom legends to plots
    custom_lines   = [Line2D([0], [0], color='k',          lw=2, linestyle='--'),
                      Line2D([0], [0], color='tab:blue',   lw=2, linestyle='-')]
    custom_lines_x = [Line2D([0], [0], color='tab:blue',   lw=2, linestyle='-'),
                      Line2D([0], [0], color='tab:orange', lw=2, linestyle='-'),
                      Line2D([0], [0], color='tab:green',  lw=2, linestyle='-'),
                      Line2D([0], [0], color='tab:red',    lw=2, linestyle='-')]
    custom_lines_d = [Line2D([0], [0], color='tab:blue',   lw=2, linestyle='-'),
                      Line2D([0], [0], color='tab:orange', lw=2, linestyle='-'),
                      Line2D([0], [0], color='tab:green',  lw=2, linestyle='-')]
    ax[0, 0].legend(custom_lines, ['Bounds', 'Controlled zone temperature'], fontsize=15, loc="best")
    ax[1, 0].legend(custom_lines_x, ['Floor temperature', 'Interior facade temperature', 'Exterior facade temperature', 'Controlled zone temperature'], fontsize=15, loc="best")
    ax[2, 0].legend(custom_lines, ['Bounds', 'Zone HVAC heat flow'], fontsize=15, loc="best")
    ax[3, 0].legend(custom_lines_x, ['Outdoor air temperature', 'Occupant heat load', 'Solar irradiance'], fontsize=15, loc="best")
                   #custom_lines_d
    fig.show()
    input("plotted solution")




ymin_traj = trajectories['ymin'].detach().reshape(nsteps_test+1, nref)
ymax_traj = trajectories['ymax'].detach().reshape(nsteps_test+1, nref)
x_traj = trajectories['x'].detach().reshape(nsteps_test+1, nx)
d_traj = trajectories['d'].detach().reshape(nsteps_test+1, nd)
u_traj = trajectories['u'].detach().reshape(nsteps_test, nu)
y_traj = trajectories['y'].detach().reshape(nsteps_test+1, ny)
plot_solution(ymin_traj,ymax_traj, x_traj,y_traj,d_traj,u_traj)

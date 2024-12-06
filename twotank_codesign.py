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


from twotank_DPC_endpt import TwoTankPredict, train_DPC


from torch.utils.data import Dataset, DataLoader

import Nets
import matplotlib.pyplot as plt
import twotank_utils

from torch.func import functional_call, vmap, grad

# Computes trajectories x, u of the control system "cl_system" under design c and target xr
def compute_traj(c, x0, xr, cl_system):
    batsize = len(x0)

    c_expand = c.unsqueeze(1).repeat(1,nsteps+1,1)
    data = {'x': x0,
            'r': xr,
            'c': c_expand}
    trajectories = cl_system(data)
    x = trajectories['x'].reshape(batsize, nsteps + 1, nx)
    u = trajectories['x'].reshape(batsize, nsteps + 1, nx)

    return x,u


def c_correction(c, x0, xr, cl_system, c_restore, alpha, n_corr_steps):

    xresid_list = []
    for _ in range(n_corr_steps):

        c = c_restore(c)

        x, u = compute_traj(c, x0, xr, cl_system)
        x_f  = x[:,-1,:]
        xr_f = xr[:,-1,:]

        grads_x = 2*(x_f - xr_f)

        xresid = torch.norm( x_f - xr_f , dim=1).mean().item()
        xresid_list.append(xresid)

        grads_c = torch.autograd.grad( x_f, c, grads_x, retain_graph = True )[0]
        c = c - alpha*grads_c

    c = c_restore(c)

    return c




seed = 2
torch.manual_seed(seed)
np.random.seed(seed)
plotting = False


"""
Instantiate the two-tank control system class and variables
"""
gt_model = psl.nonautonomous.TwoTank()
# sampling rate
ts = gt_model.params[1]['ts']
# problem dimensions
nx = gt_model.nx    # number of states
nu = gt_model.nu    # number of control inputs
nref = nx           # number of references
nc = 2              # size of c


two_tank_ode = TwoTankPredict()
integrator = integrators.RK4(two_tank_ode, h=torch.tensor(ts))  # using 4th order runge kutta integrator
model  = Node(integrator, ['x', 'u', 'c'], ['x'], name='model')

nsteps = 30                 # prediction horizon
ntraj = nx*(nsteps + 1)     # size of the reference trajectory (for NN input)
                            #    (trajectory includes nsteps+1 for initial condition)


"""
Initiate an upper-level prediction model, and generate its initial
   distribution over upper-level variables c, for DPC training
"""

c_input_size = nc
c_net = Nets.ReLUnet(c_input_size, 1, hidden_sizes = [(c_input_size),2*(c_input_size),2*(c_input_size), (c_input_size), nc], batch_norm = True, initialize = True)
sigmoid = torch.nn.Sigmoid()




"""
Instantiate some elements of the dataset: static start and end points for the control task
"""

n_train = 30000
n_dev   = 1000

x0 = torch.Tensor([0.0,0.0])

# Generate train x0 and xr
x0_train = x0*torch.ones(n_train,        1, 2)
x2 = torch.rand(n_train,1); x1 = x2*torch.rand(n_train,1)
xf_train = torch.cat((x1,x2),dim=1)
xr_train = xf_train.unsqueeze(1).repeat(1,nsteps+1,1)

# Generate dev x0 and xr
x0_dev = x0*torch.ones(n_dev,        1, 2)
x2 = torch.rand(n_dev,1); x1 = x2*torch.rand(n_dev,1)
xf_dev = torch.cat((x1,x2),dim=1)
xr_dev = xf_dev.unsqueeze(1).repeat(1,nsteps+1,1)


""" Create and pre-train the inner-loop DPC model """
umin = 0
umax = 1.
xmin = 0
xmax = 1.
net = blocks.MLP_bounds(insize=nx + nref + nc,
                        outsize=nu, hsizes=[32, 32],
                        nonlin=activations['gelu'], min=umin, max=umax)



"""  Define dataset for c: used for DPC training only!!  """
cmax = 0.10
cmin = 0.01

c = (cmax-cmin)*torch.rand(n_train,nc) + cmin
target_c_train = c.unsqueeze(1).repeat(1, nsteps+1, 1)

c = (cmax-cmin)*torch.rand(n_dev,nc) + cmin
target_c_dev = c.unsqueeze(1).repeat(1, nsteps+1, 1)

# Used in upper-level prediction and correction models
cpred_min = target_c_train.min().item()
cpred_max = target_c_train.max().item()

def c_sigmoid(c_in):
    return ( cpred_min + (cpred_max-cpred_min)*sigmoid(c_in) )

def c_project(c_in):
    return torch.clamp(c_in, min=cpred_min, max=cpred_max)



"""  Define a model to generate optimal-control traj for each  c """

target_net = blocks.MLP_bounds(insize=nx + nref + nc,
                               outsize=nu, hsizes=[64, 64],
                               nonlin=activations['gelu'], min=umin, max=umax)

target_train_data = {'x':         x0_train,
                     'r':         xr_train,
                     'c':   target_c_train }
target_train_dataset = DictDataset(target_train_data, name='train')

target_dev_data   = {'x':         x0_dev,
                     'r':         xr_dev,
                     'c':   target_c_dev }
target_dev_dataset = DictDataset(target_dev_data, name='dev')


DPC_target_epochs = 15
print("Entering training for target trajectories for {} epochs".format(DPC_target_epochs))
target_net, target_outputs = train_DPC(target_net, target_train_dataset, target_dev_dataset, DPC_target_epochs, nsteps=nsteps)
target_policy = Node(target_net, ['x', 'r', 'c'], ['u'], name='target_policy')
target_system = System([target_policy, model], nsteps=nsteps,
                   name='target_system')




"""
 Create the upper-level training dataset
"""

batch_size_ul = 800
c_optimizer = torch.optim.Adam(c_net.parameters(), lr=5e-4)
c_train_loader = DataLoader( list(zip(x0_train, xf_train)), shuffle=True, batch_size=batch_size_ul )


gamma = 0.001
corr_steps_train = 5
corr_steps_post  = 50

train_corr = 1
if train_corr:
    c_restore_train = lambda c, x0, xr: c_correction(c, x0, xr, target_system, c_project, gamma, corr_steps_train)
    c_restore_post  = lambda c, x0, xr: c_correction(c, x0, xr, target_system, c_project, gamma, corr_steps_post)
else:
    c_restore_train = lambda c, x0, xr: c_project(c)
    c_restore_post  = lambda c, x0, xr: c_correction(c, x0, xr, target_system, c_project, gamma, corr_steps_post)



"""
Upper-level training loop
"""
train_xloss_list = []
train_csum_list = []
test_xloss_list = []
test_csum_list = []

test_loss_list = []
test_ref_mean_list = []
test_cdiff_list = []
test_cabs_list = []
test_xresid_list = []
test_cresid_list = []
epochs = 80
DPC_dev_losses = []
bar_positions = [0]
for epoch in range(epochs):


    """Upper-level eval routine"""
    if True:

        (x0,xf) = (x0_dev, xf_dev)
        batsize = len(xf)

        xr = xf.unsqueeze(1).repeat(1,nsteps+1,1)

        pred_input = xf
        c = c_net(pred_input)
        c = c_restore_train( c, x0, xr )

        c_expand = c.unsqueeze(1).repeat(1,nsteps+1,1)

        data = {'x': x0,
                'r': xr,
                'c': c_expand }
        trajectories = target_system(data)
        x   = trajectories['x'].reshape(batsize, nsteps + 1, nx)

        cdiff = (c[:,0] - c[:,1]).mean()
        cabs  = (c[:,0] - c[:,1]).abs().mean()
        xresid = torch.norm( x[:,-1,:] - xr[:,-1,:] , dim=1).mean()    # target/eval only the last time frame

        xloss = torch.nn.MSELoss()( x[:,-1,:], xr[:,-1,:] )            # only match the last time frame
        csum  = c.sum(1).mean()
        test_xloss_list.append(xloss.detach().mean().item())
        test_csum_list.append(  csum.detach().mean().item())

        x_dev = x  # save for plotting at the end
        if epoch==0:
            x_dev_init = x



    """ Upper-level training of 1 epoch """
    # Training: predict c given x0,xr
    for (x0, xf) in c_train_loader:     # x0 is constant over all samples

        batsize = len(x0)
        xr = xf.unsqueeze(1).repeat(1,nsteps+1,1)

        pred_input = xf
        c = c_net(pred_input)
        c = c_restore_train( c, x0, xr )

        c_expand = c.unsqueeze(1).repeat(1,nsteps+1,1)

        data = {'x': x0,
                'r': xr,
                'c': c_expand }

        trajectories = target_system(data)
        x = trajectories['x'].reshape(batsize, nsteps + 1, nx)

        xloss = torch.nn.MSELoss()( x[:,-1,:], xr[:,-1,:] )     # only match the last time frame
        csum = c.sum(1).mean()
        closs = 1.0*csum
        cdiff = (c[:,0] - c[:,1]).mean()

        loss = xloss + closs
        loss.backward()
        c_optimizer.step()
        c_optimizer.zero_grad()

        train_xloss_list.append(xloss.detach().mean().item())
        train_csum_list.append(csum.detach().mean().item())

plt.semilogy( range(len(train_xloss_list)), train_xloss_list, label = 'xloss')
plt.semilogy( range(len(train_csum_list)),  train_csum_list,  label = 'csum')
plt.xlabel('Outer training iteration')
plt.ylabel('Training set batch loss')
plt.legend()
plt.show()


plt.semilogy( range(len(test_xloss_list)), test_xloss_list, label = 'xloss')
plt.semilogy( range(len(test_csum_list)), test_csum_list, label = 'csum')
plt.xlabel('Outer test epoch')
plt.ylabel('Test set loss')
plt.legend()
plt.show()



"""
Recompute the test set trajectories
"""
(x0,xf) = (x0_dev, xf_dev)
batsize = len(xf)

xr = xf.unsqueeze(1).repeat(1,nsteps+1,1)

pred_input = xf
c = c_net(pred_input)
#c = c_sigmoid(c)
c = c_correction(c, x0, xr, target_system, c_project, 0.0005, 5)


c_expand = c.unsqueeze(1).repeat(1,nsteps+1,1)

data = {'x': x0,
        'r': xr,
        'c': c_expand }
trajectories = target_system(data)
x   = trajectories['x'].reshape(batsize, nsteps + 1, nx)

xpred = x
cpred = c

for k in range(len(xpred)):

    plt.plot( [xf[k][0]], [xf[k][1]],  'r*', label=r"Target point" )
    plt.plot(    xpred[k][:,0].detach(),   xpred[k][:,1].detach(),  'b*-', label=r"Predicted trajectory, c = [{:.4f},  {:.4f}]".format(cpred[k][0], cpred[k][1]) )
    plt.xlim(0,1.0)
    plt.ylim(0,1.0)
    plt.legend()
    plt.show()

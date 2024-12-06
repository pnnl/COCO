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


from twotank_DPC_varc import TwoTankPredict, train_DPC


from torch.utils.data import Dataset, DataLoader

import Nets
import matplotlib.pyplot as plt
import twotank_utils

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)


"""
Instantiate the two-tank control system class and variables
"""
gt_model = psl.nonautonomous.TwoTank()
# sampling rate
ts = gt_model.params[1]['ts']
# problem dimensions
nx = gt_model.nx    # number of states
nu = gt_model.nu    # number of control inputs
nref  = nx          # number of references
nc = nx             # size of c


two_tank_ode = TwoTankPredict()
integrator = integrators.RK4(two_tank_ode, h=torch.tensor(ts))  # using 4th order runge kutta integrator
model  = Node(integrator, ['x', 'u', 'c'], ['x'], name='model')

nsteps = 50                 # prediction horizon
ntraj = nx*(nsteps + 1)     # size of the reference trajectory (for NN input)
                            #    (trajectory includes nsteps+1 for initial condition)


"""
Initiate an upper-level prediction model, and generate its initial
   distribution over upper-level variables c, for DPC training
"""
c_input_size = ntraj
c_net = Nets.ReLUnet(c_input_size, nc, hidden_sizes = [(c_input_size),2*(c_input_size),2*(c_input_size), (c_input_size), nc], batch_norm = True, initialize = True)
sigmoid = torch.nn.Sigmoid()




"""
Instantiate some elements of the dataset: static start and end points for the control task
"""
nsteps  = 50       # prediction horizon
n_train = 50000    # number of sampled scenarios
n_dev   = 1000

x0 = torch.Tensor([0.0,0.0])


# Generate training x0 and xr
x0_train = x0*torch.ones(n_train, 1, 2)
xf_train = torch.rand(n_train,2)
xr_train = xf_train.unsqueeze(1).repeat(1,nsteps+1,1)


# Generate dev x0 and xr
x0_dev = x0*torch.ones(n_dev, 1, 2)
xf_dev = torch.rand(n_dev,2)
xr_dev = xf_dev.unsqueeze(1).repeat(1,nsteps+1,1)


""" Create and pre-train the inner-loop DPC model """
umin = 0
umax = 1.
xmin = 0
xmax = 1.
net = blocks.MLP_bounds(insize=nx + nref + nc,
                        outsize=nu, hsizes=[64, 64],
                        nonlin=activations['gelu'], min=umin, max=umax)


"""  Define the ground-truth c dataset """
cmax = 0.12
cmin = 0.04

c = (cmax-cmin)*torch.rand(n_train,2) + cmin
target_c_train = c.unsqueeze(1).repeat(1, nsteps+1, 1)

c = (cmax-cmin)*torch.rand(n_dev,2) + cmin
target_c_dev = c.unsqueeze(1).repeat(1, nsteps+1, 1)


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
target_net, target_outputs, ll_model_list = train_DPC(target_net, target_train_dataset, target_dev_dataset, DPC_target_epochs)
del ll_model_list

target_policy = Node(target_net, ['x', 'r', 'c'], ['u'], name='target_policy')
target_system = System([target_policy, model], nsteps=nsteps,
                   name='target_system')

target_trajectories_train = target_system(target_train_data)
target_trajectories_dev   = target_system(target_dev_data)

target_x_train = target_trajectories_train['x'].detach()
target_x_dev   = target_trajectories_dev['x'].detach()





"""
 Create the upper-level training dataset
"""
batch_size_ul = 400
c_optimizer = torch.optim.Adam(c_net.parameters(), lr=5e-4)   #  1e-3
c_train_loader = DataLoader( list(zip(x0_train, xr_train, target_c_train, target_x_train)), shuffle=True, batch_size=batch_size_ul )

cpred_min = target_c_train.min().item()
cpred_max = target_c_train.max().item()

def c_predictor(inputs):
    return ( cpred_min + (cpred_max-cpred_min)*sigmoid(c_net(pred_input)) )




"""
Upper-level training loop
"""

test_loss_list = []
test_ref_mean_list = []
test_c1_mean_list = []
test_c2_mean_list = []
test_csum_mean_list = []
test_xresid_list = []
test_cresid_list = []
epochs = 150
DPC_dev_losses = []
bar_positions = [0]
x_dev_list = []
for epoch in range(epochs):


    """Upper-level eval routine"""
    with torch.no_grad():
        (x0, xr, ctarget, xtarget) = (x0_dev, xr_dev, target_c_dev, target_x_dev)
        batsize = len(target_x_dev)

        pred_input = xtarget.reshape(batsize,-1)    # TODO: are these squeeze / reshapes correct?
        #c = ( cmin + (cmax-cmin)*sigmoid(c_net(pred_input)) )
        c = c_predictor(pred_input)
        c = c.unsqueeze(1).repeat(1,nsteps+1,1)

        data = {'x': x0,
                'r': xr,
                'c': c }
        trajectories = target_system(data)
        x   = trajectories['x'].reshape(batsize, nsteps + 1, nx)

        cresid = torch.norm( c.reshape(batsize,-1) - ctarget.reshape(batsize,-1) , dim=1).mean()     # TODO:  Try a relative error
        xresid = torch.norm( x.reshape(batsize,-1) - xtarget.reshape(batsize,-1) , dim=1).mean()

        test_xresid_list.append( xresid.item() )
        test_cresid_list.append( cresid.item() )
        print("Outer Epoch {}".format(epoch))
        print("xresid = {}".format(xresid.item()))
        print("cresid = {}".format(cresid.item()))

        x_dev = x  # save for plotting at the end
        if epoch==0:
            x_dev_init = x
        x_dev_list.append( x_dev )


    """ Upper-level training of 1 epoch """
    # Training: predict c given x0,xr
    for (x0, xr, ctarget, xtarget) in c_train_loader:
        batsize = len(x0)

        pred_input = xtarget.reshape(len(xtarget),-1)
        c = c_predictor(pred_input)
        c = c.unsqueeze(1).repeat(1,nsteps+1,1)

        data = {'x': x0,
                'r': xr,
                'c': c }

        trajectories = target_system(data)
        x   = trajectories['x'].reshape(batsize, nsteps + 1, nx)

        closs = torch.nn.MSELoss()(c, ctarget)
        xloss = torch.nn.MSELoss()(x, xtarget)
        loss = xloss

        loss.backward()
        c_optimizer.step()
        c_optimizer.zero_grad()


plt.semilogy( range(len(test_xresid_list)),   test_xresid_list,  label=r"$\|(x - xr)\|$" )
plt.semilogy( range(len(test_cresid_list)),      test_cresid_list,     label=r"$\|(c - c^{\star})\|$" )


plt.xlabel('Outer training epoch')
plt.legend()
plt.show()



"""
Recompute the test set trajectories (code copied from eval in the loop)
"""
(x0, xr, ctarget, xtarget) = (x0_dev, xr_dev, target_c_dev, target_x_dev)
batsize = len(target_x_dev)

pred_input = xtarget.reshape(batsize,-1)
cpred = c_predictor(pred_input)
c = cpred.unsqueeze(1).repeat(1,nsteps+1,1)

data = {'x': x0,
        'r': xr,
        'c': c }
trajectories = target_system(data)   #cl_system(data)
x   = trajectories['x'].reshape(batsize, nsteps + 1, nx)


xpred = x


for k in range(len(xtarget)):
    plt.plot(  xtarget[k][:,0].detach(), xtarget[k][:,1].detach(),  'k-', label=r"Target trajectory, c = [{:.4f},  {:.4f}]".format(ctarget[k].mean(0)[0], ctarget[k].mean(0)[1]) )
    for i in range(5):
        plt.plot(    x_dev_list[i][k][:,0].detach(),   x_dev_list[i][k][:,1].detach(),  'b-', label=r"Intermediate trajectories" )
        # set axis limits
    plt.plot(    x_dev_list[-1][k][:,0].detach(),   x_dev_list[-1][k][:,1].detach(),  'g-', label=r"Predicted trajectory, c = [{:.4f},  {:.4f}]".format(cpred[k][0], cpred[k][1]) )
    # set axis limits
    plt.xlim(0,1.0)
    plt.ylim(0,1.0)
    plt.legend()
    plt.show()

# In this variation on twotank_cotrain.py,
# We generate a distribution of c2 and learn the corresponding optimal c1

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


from twotank_DPC import TwoTankPredict, train_DPC


from torch.utils.data import Dataset, DataLoader

import Nets
import matplotlib.pyplot as plt
import twotank_utils

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

nsteps = 20 #50                 # prediction horizon
ntraj = nx*(nsteps + 1)     # size of the reference trajectory (for NN input)
                            #    (trajectory includes nsteps+1 for initial condition)


"""
Initiate an upper-level prediction model, and generate its initial
   distribution over upper-level variables c, for DPC training
"""

c_predictor = Nets.ReLUnet(nx+ntraj,nc, hidden_sizes = [(nx+ntraj),2*(nx+ntraj),2*(nx+ntraj), (nx+ntraj), nc], batch_norm = True, initialize = True)
# Decide bounds on the prediction
cmin = 0.0
cmax = 0.1
sigmoid = torch.nn.Sigmoid()




"""
Construct dataset and pre-train the DPC
"""

n_samples = 2000    # number of sampled scenarios

#  sampled references for training the policy
list_refs = [torch.rand(1, 1)*torch.ones(nsteps+1, nref) for k in range(n_samples)]
ref = torch.cat(list_refs)
train_ref = ref.reshape([n_samples, nsteps+1, nref])
train_x0  = torch.rand(n_samples, 1, nx)



# Generate c
pred_input = torch.cat( (train_x0.squeeze(1), train_ref.reshape(len(train_ref),-1)), dim=1 )
train_c = ( cmin + (cmax-cmin)*sigmoid(c_predictor(pred_input))  ).detach()
train_c = train_c.unsqueeze(1).repeat(1,nsteps+1,1)
# Training dataset
train_data = DictDataset({'x': train_x0,
                          'r': train_ref,
                          'c': train_c}, name='train')

# references for dev set
list_refs = [torch.rand(1, 1)*torch.ones(nsteps+1, nref) for k in range(n_samples)]
ref = torch.cat(list_refs)
dev_ref = ref.reshape([n_samples, nsteps+1, nref])
dev_x0  = torch.rand(n_samples, 1, nx)


# Generate c
pred_input = torch.cat( (dev_x0.squeeze(1), dev_ref.reshape(len(dev_ref),-1)), dim=1 )
dev_c = ( cmin + (cmax-cmin)*sigmoid(c_predictor(pred_input)) ).detach()
dev_c = dev_c.unsqueeze(1).repeat(1,nsteps+1,1)
# Development dataset
dev_data = DictDataset({'x': dev_x0,
                        'r': dev_ref,
                        'c': dev_c}, name='dev')


umin = 0
umax = 1.
xmin = 0
xmax = 1.

"""
Instantiate the inner-loop DPC model
"""
net = blocks.MLP_bounds(insize=nx + nref + nc,
                        #insize=nx + nref,
                        outsize=nu, hsizes=[32, 32],
                        nonlin=activations['gelu'], min=umin, max=umax)


DPC_pretrain_epochs = 150
print("Entering initial DPC pretrain: {} epochs".format(DPC_pretrain_epochs))
net,outputs = train_DPC(net, train_data, dev_data, DPC_pretrain_epochs)

policy = Node(net, ['x', 'r', 'c'], ['u'], name='policy')
cl_system = System([policy, model], nsteps=nsteps,
                   name='cl_system')




"""
Generate target trajectories corresponding to c1 = 0
"""
c_target = torch.Tensor([0.0,0.04])
target_net = blocks.MLP_bounds(insize=nx + nref + nc,
                               #insize=nx + nref,
                               outsize=nu, hsizes=[32, 32],
                               nonlin=activations['gelu'], min=umin, max=umax)

target_train_data = {'x': train_x0,
                     'r': train_ref,
                     'c': c_target*torch.ones(len(train_x0), nsteps+1, 2)}

target_dev_data = {'x': dev_x0,
                   'r': dev_ref,
                   'c': c_target*torch.ones(len(dev_x0), nsteps+1, 2)}


print("Entering training for target trajectories: {} epochs".format(DPC_pretrain_epochs))
#target_net = train_DPC(target_net, target_train_data, target_dev_data, DPC_pretrain_epochs)
target_policy = Node(target_net, ['x', 'r', 'c'], ['u'], name='target_policy')
target_system = System([target_policy, model], nsteps=nsteps,
                   name='target_system')


target_trajectories_train = target_system(target_train_data)
target_trajectories_dev   = target_system(target_dev_data)

# TODO / IMPORTANT: These target trajectories are the same as in lower level,
#                   and overwrite the tank-emptying trajectories from the original experiment
#                   To retrieve the tank-emptying experiment, comment these 2 lines.
target_x_train = target_trajectories_train['x'].detach()
target_x_dev   = target_trajectories_dev['x'].detach()

print("target_x_dev")
print( target_x_dev )


################## Testing warm start  ######################
#net = train_DPC(net, train_data, dev_data, DPC_pretrain_epochs)
#net = train_DPC(net, train_data, dev_data, DPC_pretrain_epochs)
#net = train_DPC(net, train_data, dev_data, DPC_pretrain_epochs)
#input("Done with retrains")
################## Testing warm start  ######################


"""
Important: this rewrites upper-level target trajectories to be equal to lower-level ones
"""
print("target_x_dev.shape")
print( target_x_dev.shape )
print("dev_ref.shape")
print( dev_ref.shape )
target_x_train = train_ref
target_x_dev = dev_ref



"""
Create the upper-level training dataset
"""
batch_size_ul = 100
c_optimizer = torch.optim.Adam(c_predictor.parameters(), lr=5e-4)
c_train_loader = DataLoader( list(zip(train_x0,train_ref,target_x_train)), shuffle=False, batch_size=batch_size_ul )

"""
Upper-level training loop
"""

# Loss function is MSE between optimal trajectory and the empty tank flatline
#loss(x)  need to know the shape of x

# train this in neuromancer (can use the same data containers)
test_loss_list = []
test_c1_mean_list = []
test_c2_mean_list = []
test_csum_mean_list = []
epochs = 50  #100
DPC_retrain_epochs = 10
DPC_dev_losses = outputs['dev_losses_epoch']
bar_positions = [0,len(DPC_dev_losses)]
for epoch in range(epochs):


    with torch.no_grad():
        (x0, xr, xtarget) = (dev_x0, dev_ref, target_x_dev)
        batsize = len(x0)

        pred_input = torch.cat( (x0.squeeze(1), xr.reshape(len(xr),-1)), dim=1 )
        c = ( cmin + (cmax-cmin)*sigmoid(c_predictor(pred_input)) )
        c = c.unsqueeze(1).repeat(1,nsteps+1,1)
        #c = c.unsqueeze(1)*torch.ones(n_samples, nsteps+1, 2)

        data = {'x': x0,
                'r': xr,
                'c': c }

        #cl_system.nsteps = nsteps
        trajectories = cl_system(data)
        x   = trajectories['x'].reshape(batsize, nsteps + 1, nx)
        #ref = trajectories['r'].detach().reshape(batsize, nsteps + 1, nref)
        #u   = trajectories['u'].detach().reshape(batsize, nsteps, nu)

        #loss = torch.nn.MSELoss()(x, torch.zeros(x.shape))

        test_c1_mean = c.squeeze()[:,0][:,0].mean()
        test_c2_mean = c.squeeze()[:,0][:,1].mean()

        refloss = torch.nn.MSELoss()(x, xtarget)
        closs   = 0.1*(test_c1_mean + test_c2_mean)

        loss = refloss #+ closs

        test_loss_list.append( loss.item() )
        test_c1_mean_list.append( test_c1_mean.item() )
        test_c2_mean_list.append( test_c2_mean.item() )
        test_csum_mean_list.append( test_c1_mean.item() + test_c2_mean.item() )
        print("Outer Epoch {}".format(epoch))
        print("loss = {}".format(refloss.item()))
        print("c1,c2 = {},  {}".format(test_c1_mean.item(),test_c2_mean.item()))

        x_dev = x  # save for plotting at the end
        if epoch==0:
            x_dev_init = x

    # Training: predict c given x0,xr
    for (x0,xr,xtarget) in c_train_loader:
        batsize = len(x0)

        pred_input = torch.cat( (x0.squeeze(1), xr.reshape(len(xr),-1)), dim=1 )
        c = ( cmin + (cmax-cmin)*sigmoid(c_predictor(pred_input)) )
        c = c.unsqueeze(1).repeat(1,nsteps+1,1)
        #c = c.unsqueeze(1)*torch.ones(n_samples, nsteps+1, 2)

        data = {'x': x0,
                'r': xr,
                'c': c }

        #print("xtarget")
        #print( xtarget )

        #print("xr")
        #print( xr )
        #input("waiting")

        #cl_system.nsteps = nsteps
        trajectories = cl_system(data)
        x   = trajectories['x'].reshape(batsize, nsteps + 1, nx)

        #loss = torch.nn.MSELoss()(x, torch.zeros(x.shape))
        refloss = torch.nn.MSELoss()(x, xtarget)
        closs   = 0.1*(test_c1_mean + test_c2_mean)

        loss = refloss #+ closs


        loss.backward()
        c_optimizer.step()
        c_optimizer.zero_grad()



    # Rebuild DPC dataset
    """
    Do it here
    start with c_predictor inference on the whole train_data
    """

    pred_input = torch.cat( (train_x0.squeeze(1), train_ref.reshape(len(train_ref),-1)), dim=1 )
    train_c = ( cmin + (cmax-cmin)*sigmoid(c_predictor(pred_input)) ).detach()
    train_c = train_c.unsqueeze(1).repeat(1,nsteps+1,1)
    train_data = DictDataset({'x': train_x0,
                              'r': train_ref,
                              'c': train_c }, name="train")


    pred_input = torch.cat( (dev_x0.squeeze(1), dev_ref.reshape(len(dev_ref),-1)), dim=1 )
    dev_c = ( cmin + (cmax-cmin)*sigmoid(c_predictor(pred_input)) ).detach()
    dev_c = dev_c.unsqueeze(1).repeat(1,nsteps+1,1)
    dev_data =   DictDataset({'x': dev_x0,
                              'r': dev_ref,
                              'c': dev_c }, name="dev")


    # DPC network retrain
    net,outputs = train_DPC(net, train_data, dev_data, DPC_retrain_epochs, patience=5)
    policy = Node(net, ['x', 'r', 'c'], ['u'], name='policy')

    # closed-loop system model
    cl_system = System([policy, model], nsteps=nsteps,
                       name='cl_system')


    bar_positions.append( bar_positions[-1]+len(outputs['dev_losses_epoch']) )
    DPC_dev_losses += outputs['dev_losses_epoch']

plt.semilogy( range(len(test_loss_list)), test_loss_list, label="MSE(x,xr)" )
plt.semilogy( range(len(test_c1_mean_list)), test_c1_mean_list, label="c1 (mean)" )
plt.semilogy( range(len(test_c2_mean_list)), test_c2_mean_list, label="c2 (mean)" )
plt.semilogy( range(len(test_csum_mean_list)), test_csum_mean_list, label="c1 + c2 (mean)" )
plt.xlabel('Outer training epoch')
plt.legend()
plt.show()



plt.semilogy( range(len(DPC_dev_losses)),  DPC_dev_losses, label='DPC penalty loss' )
for i in range(epochs+1):  # I think this is off (black lines aren't correct); fix and look again at x resid (pink) in fig 7
    plt.axvline(x=bar_positions[i],  linestyle="--" , color='lightblue')
plt.xlabel( 'DPC training epoch' )
plt.legend()
plt.show()


pltPhase(X=x_dev_init.mean(0).detach())
pltPhase(X=x_dev.mean(0).detach())
pltPhase(X=target_x_dev.mean(0))

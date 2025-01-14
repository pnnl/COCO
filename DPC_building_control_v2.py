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




def get_data(sys, nsteps, n_samples, xmin_range, batch_size, name="train"):
    #  sampled references for training the policy
    batched_ymin = xmin_range.sample((n_samples, 1, nref)).repeat(1, nsteps + 1, 1)
    batched_ymax = batched_ymin + 2.

    # sampled disturbance trajectories from the simulation model
    batched_dist = torch.stack([torch.tensor(sys.get_D(nsteps)) for _ in range(n_samples)])

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
n_samples = 1000    # number of sampled scenarios
batch_size = 100

# range for lower comfort bound
xmin_range = torch.distributions.Uniform(18., 22.)


#train_loader, dev_loader = [
#    get_data(sys, nsteps, n_samples, xmin_range, batch_size, name=name)
#    for name in ("train", "dev")
#]
train_data, dev_data = [
    get_data(sys, nsteps, n_samples, xmin_range, batch_size, name=name)
    for name in ("train", "dev")
]
train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=train_data.collate_fn, shuffle=False)
dev_loader   = DataLoader(dev_data,   batch_size=batch_size, collate_fn=dev_data.collate_fn,   shuffle=False)


x0_dev = dev_data.datadict["x"]
d_dev = dev_data.datadict["d"]
ymax_dev = dev_data.datadict["ymax"]
ymin_dev = dev_data.datadict["ymin"]



# extract exact state space model matrices:
A = torch.tensor(sys.A)
B = torch.tensor(sys.Beta)
C = torch.tensor(sys.C)
E = torch.tensor(sys.E)

print("A")
print( A )
print("B")
print( B )
print("C")
print( C )
print("E")
print( E )


# state-space model of the building dynamics:
#   x_k+1 =  A x_k + B u_k + E d_k
xnext = lambda x, u, d: x @ A.T + u @ B.T + d @ E.T
state_model = Node(xnext, ['x', 'u', 'd'], ['x'], name='SSM')   # JK Shouldn't it be d_obs rather than d as input?
                                                                # nvm, policy takes d_obs but dynamics take d

#   y_k = C x_k
ynext = lambda x: x @ C.T
output_model = Node(ynext, ['x'], ['y'], name='y=Cx')

# partially observable disturbance model
dist_model = lambda d: d[:, d_idx]
dist_obs = Node(dist_model, ['d'], ['d_obs'], name='dist_obs')



# neural net control policy
net = blocks.MLP_bounds(
    insize=ny + 2*nref + nd_obs,
    outsize=nu,
    hsizes=[32, 32],
    nonlin=nn.GELU,
    min=umin,
    max=umax,
)
policy = Node(net, ['y', 'ymin', 'ymax', 'd_obs'], ['u'], name='policy')


# closed-loop system model
cl_system = System([dist_obs, policy, state_model, output_model],
                    nsteps=nsteps,
                    name='cl_system')
#cl_system.show()



# variables
y = variable('y')
u = variable('u')
ymin = variable('ymin')
ymax = variable('ymax')


Q_u = 0.01
Q_du = 0.1
Q_ymin = 50.0
Q_ymax = 50.0

# objectives
action_loss = 0.01 * (u == 0.0)  # energy minimization
du_loss = 0.1 * (u[:,:-1,:] - u[:,1:,:] == 0.0)  # delta u minimization to prevent agressive changes in control actions

# thermal comfort constraints
state_lower_bound_penalty = 50.*(y > ymin)
state_upper_bound_penalty = 50.*(y < ymax)

# objectives and constraints names for nicer plot
action_loss.name = 'action_loss'
du_loss.name = 'du_loss'
state_lower_bound_penalty.name = 'x_min'
state_upper_bound_penalty.name = 'x_max'

# list of constraints and objectives
objectives = [action_loss, du_loss]
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
    custom_lines = [Line2D([0], [0], color='k', lw=2, linestyle='--'),
                        Line2D([0], [0], color='tab:blue', lw=2, linestyle='-')]
    custom_lines_x = [Line2D([0], [0], color='tab:blue', lw=2, linestyle='-'),
                        Line2D([0], [0], color='tab:orange', lw=2, linestyle='-'),
                        Line2D([0], [0], color='tab:green', lw=2, linestyle='-'),
                        Line2D([0], [0], color='tab:red', lw=2, linestyle='-')]
    custom_lines_d = [Line2D([0], [0], color='tab:blue', lw=2, linestyle='-'),
                        Line2D([0], [0], color='tab:orange', lw=2, linestyle='-'),
                        Line2D([0], [0], color='tab:green', lw=2, linestyle='-')]
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

# Prediction horizon
N = nsteps_test

# Bounds
xminc = 0
xmaxc = 0

# Define problem
xc = cv.Variable((N + 1, nx))
yc = cv.Variable((N + 1, ny))
uc = cv.Variable((N, nu))
slack_lower = cv.Variable((N + 1, ny))
slack_upper = cv.Variable((N + 1, ny))


Ad = np.array(A)
Bd = np.array(B)
Ed = np.array(E)
Cd = np.array(C)
Ix = np.identity(nx)
Iy = np.identity(ny)
Iu = np.identity(nu)

uminc = np.array(umin)
umaxc = np.array(umax)

x0    = cv.Parameter(nx)
dp    = cv.Parameter((N+1, nd))
yminp = cv.Parameter((N+1, ny))
ymaxp = cv.Parameter((N+1, ny))
objective = 0
constraints = [xc[0, :] == x0]
for k in range(N):
    constraints += [xc[k + 1,:] == Ad @ xc[k,:] + Bd @ uc[k,:] + Ed @ dp[k,:]]  # + G     # should be dc[sys.d_idx, k] ?   what is d, variable or parameter? does it need a k index?
    objective   += (  cv.quad_form(uc[k,:], Q_u*Iu)       )   #Q_u*(  cv.quad_form(uc[k,:], Iu)       )
for k in range(N+1):
    constraints += [yc[k    ,:] == Cd @ xc[k,:]]  # + F + y_ss
    objective   += (  cv.quad_form(slack_upper[k,:], Q_ymax*Iy)  )
    objective   += (  cv.quad_form(slack_lower[k,:], Q_ymin*Iy)  )

constraints += [uminc <= uc   ]
constraints += [uc    <= umaxc]
constraints += [yc >= yminp]
constraints += [yc <= ymaxp]
#constraints += [yc >= yminp - slack_lower]
#constraints += [yc <= ymaxp + slack_upper]
#constraints += [slack_lower == 0]
#constraints += [slack_upper == 0]
#constraints += [yc[N,:] >= yminp - slack_lower]
#constraints += [yc[N,:] <= ymaxp + slack_upper]
prob = cv.Problem(cv.Minimize(objective), constraints)


cvxlayer_pre = CvxpyLayer(prob, parameters=[x0,dp,yminp,ymaxp], variables=[xc,yc,uc,slack_upper,slack_lower])
def cvxlayer(x0,dp,yminp,ymaxp):
    out = cvxlayer_pre(x0,dp,yminp,ymaxp, solver_args={"max_iters":100000})
    #out = cvxlayer_pre(x0,dp,yminp,ymaxp, solver_args={"solve_method":"ECOS","max_iters":10000,"abstol":1e-3,"reltol":1e-3})
    return out#[0]#, out[1]


print("x0_test.shape")
print( x0_test.shape )
print("d_test.shape")
print( d_test.shape )
print("ymin_test.shape")
print( ymin_test.shape )
print("ymax_test.shape")
print( ymax_test.shape )

out = cvxlayer(x0_test[:,0,:][0], d_test[0], ymin_test[0], ymax_test[0])
xc = out[0]
yc = out[1]
uc = out[2]
slack_upper = out[3]
slack_lower = out[4]


print("yc")
print( yc )
print("ymin_test[0]")
print( ymin_test[0] )
print("yc - ymin_test[0]")
print( yc - ymin_test[0] )
print("slack_upper")
print( slack_upper )
print("slack_lower")
print( slack_lower )


ymin_traj = ymin_test.reshape(nsteps_test+1, nref)
ymax_traj = ymax_test.reshape(nsteps_test+1, nref)
d_traj    = d_test.detach().reshape(nsteps_test+1, nd)
x_traj    = xc.detach().reshape(nsteps_test+1, nx)
u_traj    = uc.detach().reshape(nsteps_test, nu)
y_traj    = yc.detach().reshape(nsteps_test+1, ny)
plot_solution(ymin_traj,ymax_traj, x_traj,y_traj,d_traj,u_traj)

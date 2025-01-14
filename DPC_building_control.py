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

from cvxpylayers.torch.cvxpylayer import CvxpyLayer
import cvxpy as cv

# ground truth system model
sys = psl.systems['LinearSimpleSingleZone']()

# problem dimensions
nx = sys.nx           # number of states
nu = sys.nu           # number of control inputs
nd = sys.nD           # number of disturbances
nd_obs = sys.nD_obs   # number of observable disturbances
ny = sys.ny           # number of controlled outputs
nref = ny             # number of references

# control action bounds
umin = torch.tensor(sys.umin)
umax = torch.tensor(sys.umax)

def get_data(sys, nsteps, n_samples, xmin_range, batch_size, name="train"):
    #  sampled references for training the policy
    batched_xmin = xmin_range.sample((n_samples, 1, nref)).repeat(1, nsteps + 1, 1)
    batched_xmax = batched_xmin + 2.

    # sampled disturbance trajectories from the simulation model
    batched_dist = torch.stack([torch.tensor(sys.get_D(nsteps)) for _ in range(n_samples)])

    # sampled initial conditions
    batched_x0 = torch.stack([torch.tensor(sys.get_x0()).unsqueeze(0) for _ in range(n_samples)])

    data = DictDataset(
        {"x": batched_x0,
         "y": batched_x0[:,:,[-1]],
         "ymin": batched_xmin,
         "ymax": batched_xmax,
         "d": batched_dist},
        name=name,
    )

    return DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn, shuffle=False)


nsteps = 100  # prediction horizon
n_samples = 1000    # number of sampled scenarios
batch_size = 100

# range for lower comfort bound
xmin_range = torch.distributions.Uniform(18., 22.)

train_loader, dev_loader = [
    get_data(sys, nsteps, n_samples, xmin_range, batch_size, name=name)
    for name in ("train", "dev")
]





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
dist_model = lambda d: d[:, sys.d_idx]
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
    epochs=200, # TODO
    train_metric='train_loss',
    eval_metric='dev_loss',
    warmup=200,
)



# Train control policy
best_model = trainer.train()
# load best trained model
trainer.model.load_state_dict(best_model)





nsteps_test = 2000

# generate reference
np_refs = psl.signals.step(nsteps_test+1, 1, min=18., max=22., randsteps=5)
ymin_val = torch.tensor(np_refs, dtype=torch.float32).reshape(1, nsteps_test+1, 1)
ymax_val = ymin_val+2.0
# generate disturbance signal
torch_dist = torch.tensor(sys.get_D(nsteps_test+1)).unsqueeze(0)
# initial data for closed loop simulation
x0 = torch.tensor(sys.get_x0()).reshape(1, 1, nx)
data = {'x': x0,
        'y': x0[:, :, [-1]],
        'ymin': ymin_val,
        'ymax': ymax_val,
        'd': torch_dist}
cl_system.nsteps = nsteps_test
# perform closed-loop simulation
trajectories = cl_system(data)



from matplotlib.lines import Line2D
# constraints bounds
Umin = umin * np.ones([nsteps_test, nu])
Umax = umax * np.ones([nsteps_test, nu])
Ymin = trajectories['ymin'].detach().reshape(nsteps_test+1, nref)
Ymax = trajectories['ymax'].detach().reshape(nsteps_test+1, nref)
# plot closed loop trajectories
fig, ax = pltCL(Y=trajectories['y'].detach().reshape(nsteps_test+1, ny),
        R=Ymax,
        X=trajectories['x'].detach().reshape(nsteps_test+1, nx),
        D=trajectories['d'].detach().reshape(nsteps_test+1, nd),
        U=trajectories['u'].detach().reshape(nsteps_test, nu),
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



fig.show()


input("Quitting here")
quit()






# Prediction horizon
N = nsteps

# Bounds
xminc = 0
xmaxc = 0

# Define problem
xc = cv.Variable((nx, N + 1))
yc = cv.Variable((ny, N + 1))
uc = cv.Variable((nu, N))

Ad = np.array(A)
Bd = np.array(B)
Ed = np.array(E)
dc = np.array(d)
Cd = np.array(C)


print("Ad.shape")
print( Ad.shape )
print("Bd.shape")
print( Bd.shape )
print("Ed.shape")
print( Ed.shape )
print("dc.shape")
print( dc.shape )
print("Cd.shape")
print( Cd.shape )

print("nx")
print( nx )
print("ny")
print( ny )
print("nu")
print( nu )
print("nd")
print( nd )

xrp    = cv.Parameter(nx)
x_init = cv.Parameter(nx)
objectivec = 0
constraintsc = [xc[:, 0] == x_init]
for k in range(N):
    #objectivec   += cv.quad_form(xc[:, k] - xrc, Qs) + cv.quad_form(uc[:, k], Rs)
    objectivec   += cv.quad_form(xc[:, k] - xrp, Ix)
    constraintsc += [xc[:, k + 1] == Ad @ xc[:, k] + Bd @ uc[:, k] + Ed @ dc[:, k]]  # + G     # should be dc[sys.d_idx, k] ?   what is d, variable or parameter? does it need a k index?
    constraintsc += [yc[:, k + 1] == Cd @ yc[:, k]]  # + F + y_ss
    constraintsc += [yc[:, k]>=yminc, yc[:, k] <= ymaxc]
    constraintsc += [uminc <= uc[:, k], uc[:, k] <= umaxc]
#constraintsc += [ xc[:, k+1]>=xminc, xc[:, k+1] <= xmaxc]
#objectivec += cv.quad_form(xc[:, N] - xrc, QNs)
objectivec += cv.quad_form(xc[:, N] - xrp, Ix)
prob = cv.Problem(cv.Minimize(objectivec), constraintsc)



qp_cvxlayer_pre = CvxpyLayer(prob, parameters=[x_init,xrp], variables=[xc,uc])
def qp_cvxlayer(x_init,xrp):
    out = qp_cvxlayer_pre(x_init,xrp, solver_args={"solve_method":"ECOS","max_iters":10000,"abstol":1e-3,"reltol":1e-3})
    return out[0], out[1]

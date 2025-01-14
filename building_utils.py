import numpy as np
import scipy as sp

from scipy import sparse
from pylab import *
import time
import neuromancer.psl as psl

from cvxpylayers.torch.cvxpylayer import CvxpyLayer
import torch

from neuromancer.plot import pltCL


def gen_building_data_single(n_samples, nsteps, nodist=True):
    torch.manual_seed(0)
    np.random.seed(0)
    sys = psl.systems['LinearSimpleSingleZone'](seed=0)
    batched_ymin = torch.stack([torch.tensor(psl.signals.beta_walk_max_step(nsteps + 1, 1, min=18., max=22., max_step = 3.0, p = 0.1)) for _ in range(n_samples)])
    batched_ymax = batched_ymin + 2.
    batched_dist = torch.stack([torch.tensor(sys.get_D(nsteps+1)) for _ in range(n_samples)])*(0.0 if nodist else 1.0)   # JK: mult by 0 to turn off disturbances
    batched_x0   = torch.stack([torch.tensor(sys.get_x0()).unsqueeze(0) for _ in range(n_samples)])
    batched_x0[:,:,-1] = (batched_ymax[:,0,:] + batched_ymin[:,0,:]) / 2
    return batched_ymin, batched_ymax, batched_dist, batched_x0


# input matrix A is assumed to be repeated block diagonal
def link_zones_long(A, nzones, nx, dx):
    for k in range(nzones-1):
        A[(k+1)*nx+1, k*nx+1] += dx
        A[k*nx+1, (k+1)*nx+1] += dx
    for k in range(nzones):
        A[k*nx+1, k*nx+1] -= dx  if (k == 0) or (k == nzones-1) else 2*dx
    return A





if __name__ == "__main__":
    sys = psl.systems['LinearSimpleSingleZone'](seed=0)

    nzones = 3

    Asingle =  torch.Tensor(sys.A)  #torch.ones(3,3)
    print("Asingle")
    print( Asingle )
    torch.set_printoptions(precision=3, threshold=None, sci_mode=False)
    nx = 4 #3
    dx = 0.05
    A = torch.block_diag(  *tuple([Asingle    for _ in range(nzones)])  )
    print("A = ")
    print( A    )
    A = link_zones_long(A, nzones, nx, dx)
    print("A = ")
    print( A    )

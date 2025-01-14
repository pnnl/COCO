import cvxpy as cp
from cvxpy import *
import numpy as np
import scipy as sp

from scipy import sparse
from pylab import *
import time
import neuromancer.psl as psl

from cvxpylayers.torch.cvxpylayer import CvxpyLayer
import torch


from neuromancer.plot import pltCL
import building_utils
import pickle
import os

def get_building_MPC_layer(N,nu,nx,ny,nd, umin,umax, A,C,E,F,G,y_ss, Q_weight,R_weight):


    Q = Q_weight * sparse.eye(ny)
    QN = Q
    R = R_weight * sparse.eye(nu)


    # Define problem
    u = Variable((N,  nu))
    x = Variable((N+1,nx))
    y = Variable((N+1,ny))

    B = Parameter((nx,nu))

    d = Parameter((N+1,nd))
    slack_lower = Variable((N+1,ny))
    slack_upper = Variable((N+1,ny))
    x_init = Parameter(nx)
    ymin   = Parameter((N+1,ny))
    ymax   = Parameter((N+1,ny))
    objective = 0
    constraints  = [x[0,:] == x_init]

    for k in range(N):
        constraints += [x[k+1,:] == A @ x[k,:] + B @ u[k,:] + E @ d[k,:] + G]
        constraints += [u[k,:] <= umax]
        constraints += [u[k,:] >= umin]
        objective += quad_form(u[k,:],R)
    for k in range(N+1):
        constraints += [y[k,:] == C @ x[k,:] + F - y_ss]
        #constraints += [y[k,:] >= ymin[k,:] - slack_lower[k,:]]
        #constraints += [y[k,:] <= ymax[k,:] + slack_upper[k,:]]
        #constraints += [slack_lower[k,:] >= 0]
        #constraints += [slack_upper[k,:] >= 0]
        objective += quad_form(slack_upper[k,:], QN) + quad_form(slack_lower[k,:], QN)

    constraints += [y >= ymin - slack_lower]
    constraints += [y <= ymax + slack_upper]
    constraints += [slack_lower >= 0]
    constraints += [slack_upper >= 0]

    prob = Problem(Minimize(objective), constraints)


    #cvxlayer_pre = CvxpyLayer(prob, parameters=[x_init,y_init,d,ymin,ymax], variables=[u,x,y,slack_lower,slack_upper])
    #def cvxlayer(x_init,y_init,d,ymin,ymax):
    #    out = cvxlayer_pre(x_init,y_init,d,ymin,ymax)
    #    return out

    return CvxpyLayer(prob, parameters=[B,x_init,d,ymin,ymax], variables=[u,x,y,slack_lower,slack_upper])





def get_building_MPC_layer_hard(N,nu,nx,ny,nd, umin,umax, A,C,E,F,G,y_ss, R_weight):


    Q = Q_weight * sparse.eye(ny)
    QN = Q
    R = R_weight * sparse.eye(nu)


    # Define problem
    u = Variable((N,  nu))
    x = Variable((N+1,nx))
    y = Variable((N+1,ny))

    B = Parameter((nx,nu))

    d = Parameter((N+1,nd))
    slack_lower = Variable((N+1,ny))
    slack_upper = Variable((N+1,ny))
    x_init = Parameter(nx)
    ymin   = Parameter((N+1,ny))
    ymax   = Parameter((N+1,ny))
    objective = 0
    constraints  = [x[0,:] == x_init]

    for k in range(N):
        constraints += [x[k+1,:] == A @ x[k,:] + B @ u[k,:] + E @ d[k,:] + G]
        constraints += [u[k,:] <= umax]
        constraints += [u[k,:] >= umin]
        objective += quad_form(u[k,:],R)
    for k in range(N+1):
        constraints += [y[k,:] == C @ x[k,:] + F - y_ss]
        #constraints += [y[k,:] >= ymin[k,:] - slack_lower[k,:]]
        #constraints += [y[k,:] <= ymax[k,:] + slack_upper[k,:]]
        #constraints += [slack_lower[k,:] >= 0]
        #constraints += [slack_upper[k,:] >= 0]

    constraints += [y >= ymin - slack_lower]
    constraints += [y <= ymax + slack_upper]
    constraints += [slack_lower == 0]
    constraints += [slack_upper == 0]

    prob = Problem(Minimize(objective), constraints)


    #cvxlayer_pre = CvxpyLayer(prob, parameters=[x_init,y_init,d,ymin,ymax], variables=[u,x,y,slack_lower,slack_upper])
    #def cvxlayer(x_init,y_init,d,ymin,ymax):
    #    out = cvxlayer_pre(x_init,y_init,d,ymin,ymax)
    #    return out

    return CvxpyLayer(prob, parameters=[B,x_init,d,ymin,ymax], variables=[u,x,y,slack_lower,slack_upper])





def plot_solution(ymin_traj,ymax_traj, x_traj,y_traj,d_traj,u_traj):
    # constraints bounds
    Umin = umin * np.ones([nsteps, nu])
    Umax = umax * np.ones([nsteps, nu])
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
    input("Plotting solution")








if __name__ == "__main__":


    torch.manual_seed(0)
    np.random.seed(0)

    # ground truth system model
    sys = psl.systems['LinearSimpleSingleZone'](seed=0)
    nzones = 1

    # problem dimensions
    nx = nzones*sys.nx                # number of states
    nu = nzones*sys.nu                # number of control inputs
    nd = nzones*sys.nD                # number of disturbances
    ny = nzones*sys.ny                # number of controlled outputs
    nref = ny                  # number of references
    partial_observe = False
    if partial_observe:        # Toggle partial observability of the disturbance
        d_idx = sys.d_idx
    else:
        d_idx = range(nd)
    nd_obs = len(d_idx)   # number of observable disturbances
    nB = nu*nx  # This is the size of B as passed in to the policy network, should be the number of learnable elements

    # extract exact state space model matrices:
    A = torch.block_diag(  *tuple([torch.tensor(sys.A)    for _ in range(nzones)])  )
    B = torch.block_diag(  *tuple([torch.tensor(sys.Beta) for _ in range(nzones)])  )
    C = torch.block_diag(  *tuple([torch.tensor(sys.C)    for _ in range(nzones)])  )
    E = torch.block_diag(  *tuple([torch.tensor(sys.E)    for _ in range(nzones)])  )
    F = torch.zeros(ny)
    G = torch.zeros(nx)
    y_ss = torch.zeros(ny)

    # control action bounds
    #umin = torch.Tensor([sys.umin.item() for _ in range(nzones)])
    umax = torch.Tensor([sys.umax.item() for _ in range(nzones)])

    B_scale_factor = umax.max()
    umax = umax / B_scale_factor    # JK rescaling reduces computation time dramatically
    umin = -umax
    B = B * B_scale_factor          #   however, I haven't been able to verify equivalence, due to either solver instability when R!=0 or nonunique solutions when R=0

    B_mask = (B != 0.0).float()

    nsteps    = 50
    n_samples = 3

    # generate data for a single-zone building
    filename = "./data/dist_zone_data_{}steps_{}samples.p".format(nsteps, n_samples)
    if not os.path.isfile(filename):
        zone_data = building_utils.gen_building_data_single(n_samples, nsteps, nodist = False)
        pickle.dump(zone_data,open(filename,'wb'))
    else:
        zone_data = pickle.load(open(filename,'rb'))


    batched_ymin, batched_ymax, batched_dist, batched_x0 = zone_data

    # duplicate data across multiple zones
    x0   = batched_x0.repeat(1,1,nzones)
    y0   = batched_x0[:,:,[-1]].repeat(1,1,nzones)
    ymin = batched_ymin.repeat(1,1,nzones)
    ymax = batched_ymax.repeat(1,1,nzones)
    d    = batched_dist.repeat(1,1,nzones)

    B    = B.repeat(n_samples, 1, 1)


    print("sys.E" )
    print( sys.E )
    print("d" )
    print( d )
    print("d.shape" )
    print( d.shape )
    input("waiting")


    Q_weight = 5000.0
    R_weight =  1.0
    diff_solver      = get_building_MPC_layer(     nsteps,nu,nx,ny,nd, umin,umax, A,C,E,F,G,y_ss, Q_weight, R_weight)
    diff_solver_hard = get_building_MPC_layer_hard(nsteps,nu,nx,ny,nd, umin,umax, A,C,E,F,G,y_ss,           R_weight)
    solver_out  = diff_solver(B, x0.squeeze(1), d, ymin, ymax)
    u = solver_out[0]
    x = solver_out[1]
    y = solver_out[2]
    slack_upper = solver_out[3]
    slack_lower = solver_out[4]

    print("x.shape" )
    print( x.shape  )
    print("u.shape" )
    print( u.shape  )
    print("y.shape" )
    print( y.shape  )

    print("x" )
    print( x  )
    print("u" )
    print( u  )
    print("y" )
    print( y  )

    for i in range(n_samples):
        plot_solution(ymin[i], ymax[i], x[i], y[i], d[i], u[i])

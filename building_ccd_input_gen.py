import numpy as np
from datetime import date
import math
file_name = "building_ccd.in"
file=open(file_name, 'w')

ntrain_list = [15000,10000]
ntest_list = [1000]
lr_list = [1e-4,1e-3,1e-2]
nsteps_list = [50]
nzones_list = [1,2,3,4]
penalty_list = [10.0,100.0,1000.0]
alpha_list = [1e-6,1e-5,1e-4]

index = 0
for ntrain in ntrain_list:
    for ntest in ntest_list:
        for lr in lr_list:
            for nsteps in nsteps_list:
                for nzones in nzones_list:
                    for penalty in penalty_list:
                        for alpha in alpha_list:
                            print(  ntrain,
                                    ntest,
                                    lr,
                                    nsteps,
                                    nzones,
                                    penalty,
                                    alpha,
                                    index,
                                    sep=' ',
                                    file=file     )
                            index += 1
file.close()
print("Done")

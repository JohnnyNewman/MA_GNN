# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 15:03:37 2020

@author: Nils
"""


from multiprocessing import Pool


import os, sys, shutil
from optparse import OptionParser

sys.path.append(os.environ["SU2_RUN"])
import SU2

from numpy import random

from torch.quasirandom import SobolEngine

import copy

import pickle

# import cProfile

# -------------------------------------------------------------------
#  Main
# -------------------------------------------------------------------


def main():

    parser = OptionParser()
    parser.add_option(
        "-f",
        "--file",
        dest="filename",
        help="read config from FILE",
        metavar="FILE",
        default="turb_SA_RAE2822.cfg",
    )
    parser.add_option(
        "-r",
        "--name",
        dest="projectname",
        default="",
        help="try to restart from project file NAME",
        metavar="NAME",
    )
    parser.add_option(
        "-n",
        "--partitions",
        dest="partitions",
        default=1,
        help="number of PARTITIONS",
        metavar="PARTITIONS",
    )
    parser.add_option(
        "-g",
        "--gradient",
        dest="gradient",  # default="DISCRETE_ADJOINT",
        help="Method for computing the GRADIENT (CONTINUOUS_ADJOINT, DISCRETE_ADJOINT, FINDIFF, NONE)",
        metavar="GRADIENT",
        default="CONTINUOUS_ADJOINT",
    )
    parser.add_option(
        "-o",
        "--optimization",
        dest="optimization",
        default="SLSQP",
        help="OPTIMIZATION techique (SLSQP, CG, BFGS, POWELL)",
        metavar="OPTIMIZATION",
    )
    parser.add_option(
        "-q",
        "--quiet",
        dest="quiet",
        default="True",
        help="True/False Quiet all SU2 output (optimizer output only)",
        metavar="QUIET",
    )
    parser.add_option(
        "-z",
        "--zones",
        dest="nzones",
        default="1",
        help="Number of Zones",
        metavar="ZONES",
    )

    (options, args) = parser.parse_args()

    # process inputs
    options.partitions = int(options.partitions)
    options.quiet = options.quiet.upper() == "TRUE"
    options.gradient = options.gradient.upper()
    options.nzones = int(options.nzones)

    print("hello")
    print(f"filename: {options.filename}")

    run(options)


def run(options):

    # pr = cProfile.Profile()
    # pr.enable()

    # Config
    config = SU2.io.Config(options.filename)
    config.NUMBER_PART = options.partitions
    config.NZONES = int(options.nzones)
    if options.quiet:
        config.CONSOLE = "CONCISE"
    config.GRADIENT_METHOD = options.gradient

    its = int(config.OPT_ITERATIONS)  # number of opt iterations
    bound_upper = float(
        config.OPT_BOUND_UPPER
    )  # variable bound to be scaled by the line search
    bound_lower = float(
        config.OPT_BOUND_LOWER
    )  # variable bound to be scaled by the line search
    relax_factor = float(config.OPT_RELAX_FACTOR)  # line search scale
    gradient_factor = float(
        config.OPT_GRADIENT_FACTOR
    )  # objective function and gradient scale
    def_dv = config.DEFINITION_DV  # complete definition of the desing variable
    n_dv = sum(def_dv["SIZE"])  # number of design variables
    accu = float(config.OPT_ACCURACY) * gradient_factor  # optimizer accuracy
    x0 = [0.0] * n_dv  # initial design
    xb_low = [
        float(bound_lower) / float(relax_factor)
    ] * n_dv  # lower dv bound it includes the line search acceleration factor
    xb_up = [
        float(bound_upper) / float(relax_factor)
    ] * n_dv  # upper dv bound it includes the line search acceleration fa
    xb = list(zip(xb_low, xb_up))  # design bounds

    # State
    state = SU2.io.State()
    state.find_files(config)

    # add restart files to state.FILES
    if (
        config.get("TIME_DOMAIN", "NO") == "YES"
        and config.get("RESTART_SOL", "NO") == "YES"
        and gradient != "CONTINUOUS_ADJOINT"
    ):
        restart_name = config["RESTART_FILENAME"].split(".")[0]
        restart_filename = (
            restart_name + "_" + str(int(config["RESTART_ITER"]) - 1).zfill(5) + ".dat"
        )
        if not os.path.isfile(
            restart_filename
        ):  # throw, if restart files does not exist
            sys.exit("Error: Restart file <" + restart_filename + "> not found.")
        state["FILES"]["RESTART_FILE_1"] = restart_filename

        # use only, if time integration is second order
        if config.get("TIME_MARCHING", "NO") == "DUAL_TIME_STEPPING-2ND_ORDER":
            restart_filename = (
                restart_name
                + "_"
                + str(int(config["RESTART_ITER"]) - 2).zfill(5)
                + ".dat"
            )
            if not os.path.isfile(
                restart_filename
            ):  # throw, if restart files does not exist
                sys.exit("Error: Restart file <" + restart_filename + "> not found.")
            state["FILES"]["RESTART_FILE_2"] = restart_filename

    # Project

    if os.path.exists(options.projectname):
        project = SU2.io.load_data(options.projectname)
        project.config = config
    else:
        project = SU2.opt.Project(config, state)

    print(project)
    # print(config)

    n_dv = len(project.config["DEFINITION_DV"]["KIND"])
    project.n_dv = n_dv

    soboleng = SobolEngine(n_dv + 3, True)

    n_samples = 1024

    X_draw = soboleng.draw(n_samples).numpy()

    # X_draw = random.rand(n_samples, n_dv+2)

    Ma_upper = 0.95
    Ma_lower = 0.05

    alpha_upper = 20.0
    alpha_lower = 0.0

    Re_upper = 1.0e07
    Re_lower = 1.0e06

    Ma_mat = X_draw[:, 0] * (Ma_upper - Ma_lower) + Ma_lower
    AOA_mat = X_draw[:, 1] * (alpha_upper - alpha_lower) + alpha_lower
    Re_mat = X_draw[:, 2] * (Re_upper - Re_lower) + Re_lower

    dv_scale = 1.0e-04
    dv_mat = dv_scale * (X_draw[:, 3:] - 0.5)

    dv_list = []
    # Ma_list = []
    # alpha_list
    for i in range(n_samples):
        dv_list.append(dv_mat[i].tolist())
        # alpha = double(X[i,1])
        # Ma = double(X[i,0])

    print(dv_list[:3])
    pickle.dump(dv_list, open("dv_list.p", "wb"))
    pickle.dump(Ma_mat, open("Ma_mat.p", "wb"))
    pickle.dump(AOA_mat, open("AOA_mat.p", "wb"))
    pickle.dump(Re_mat, open("Re_mat.p", "wb"))

    # inputs = [(copy.deepcopy(project), dvs) for dvs in dv_list]

    print(len(dv_list))

    inputs = []
    for i, dvs in enumerate(dv_list):
        # dsn = project.new_design(project.config)
        config = copy.deepcopy(project.config)
        config.MACH_NUMBER = Ma_mat[i]
        config.AOA = AOA_mat[i]
        config.REYNOLDS_NUMBER = Re_mat[i]
        dsn = SU2.eval.design.Design(config, folder="DESIGNS/DSN_{:04d}".format(i))
        inputs.append({"design": dsn, "dvs": dvs})

    with Pool(processes=16) as pool:
        outputs = pool.map(run_one_design, inputs)

    print(outputs)

    return


from scipy.optimize import NonlinearConstraint, Bounds, minimize


def run_design_optimization(inputs, its, accu):

    design = inputs["design"]
    dvs = inputs["dvs"]

    print("optimizing", design.__repr__())

    nlc_cieq = NonlinearConstraint(
        design.con_cieq, 0.0, np.inf, jac=design.con_dcieq, keep_feasible=True
    )
    nlc_ceq = NonlinearConstraint(
        design.con_ceq, 0.0, 0.0, jac=design.con_dieq, keep_feasible=True
    )
    # bounds = Bounds(lb=, ub=, keep_feasible=True)

    # solver_options = {'gtol': 1e-05, 'norm': inf, 'maxiter': its, 'disp': True, 'return_all': False, 'finite_diff_rel_step': None})
    # minimize(fun=design.obj_f, x0=dvs, method="BFGS", jac=design.obj_df, bounds=(), constraints=(nlc_cieq, nlc_ceq), tol=, options=solver_options)


def run_one_design(inputs):

    design = inputs["design"]
    dvs = inputs["dvs"]

    print("running", design.__repr__())

    # project = copy.deepcopy(project)

    # design = project.new_design(project.config)

    x = dvs

    obj_f = design.obj_f(x)
    print(design.__repr__(), "obj_f:", obj_f)

    con_ceq = design.con_ceq(x)
    print(design.__repr__(), "con_ceq:", con_ceq)

    con_cieq = design.con_cieq(x)
    print(design.__repr__(), "con_cieq:", con_cieq)

    obj_df = design.obj_df(x)
    print(design.__repr__(), "obj_df:", obj_df)

    con_dceq = design.con_dceq(x)
    print(design.__repr__(), "con_dceq:", con_dceq)

    con_dcieq = design.con_dcieq(x)
    print(design.__repr__(), "con_dcieq:", con_dcieq)

    return dvs, obj_f, con_ceq, con_cieq, obj_df, con_dceq, con_dcieq

    # for i in range(1000):
    #     print(f"iter {i}")
    #     design = project.new_design(config)

    #     dvs = random.rand(n_dv)*2.e-05 - 1.e-05
    #     print(dvs)
    #     #project.obj_f(dvs.tolist())
    #     #config.unpack_dvs(dvs.tolist())

    #     print('DV_VALUE_NEW', config['DV_VALUE_NEW'])
    #     print('DV_VALUE_OLD', config['DV_VALUE_OLD'])
    #     #vals = SU2.eval.aerodynamics(config)

    #     x = dvs

    #     obj_f = project.obj_f(x)
    #     print("obj_f:", obj_f)

    #     con_ceq       = project.con_ceq(x)
    #     print("con_ceq:", con_ceq)

    #     con_cieq      = project.con_cieq(x)
    #     print("con_cieq:", con_cieq)

    #     obj_df  = project.obj_df(x)
    #     print("obj_df:", obj_df)

    #     con_dceq         = project.con_dceq(x)
    #     print("con_dceq:", con_dceq)

    #     con_dcieq = project.con_dcieq(x)
    #     print("con_dcieq:", con_dcieq)

    #     # pr.dump_stats(f"test_run_iter{i}.prof")

    # func           = obj_f
    #     f_eqcons       = con_ceq
    #     f_ieqcons      = con_cieq

    #     # gradient handles
    #     if project.config.get('GRADIENT_METHOD','NONE') == 'NONE':
    #         fprime         = None
    #         fprime_eqcons  = None
    #         fprime_ieqcons = None
    #     else:
    #         fprime         = obj_df
    #         fprime_eqcons  = con_dceq
    #         fprime_ieqcons = con_dcieq

    # # Optimize
    # if optimization == 'SLSQP':
    #   SU2.opt.SLSQP(project,x0,xb,its,accu)
    # if optimization == 'CG':
    #   SU2.opt.CG(project,x0,xb,its,accu)
    # if optimization == 'BFGS':
    #   SU2.opt.BFGS(project,x0,xb,its,accu)
    # if optimization == 'POWELL':
    #   SU2.opt.POWELL(project,x0,xb,its,accu)

    # pr.disable()

    # # rename project file
    # if projectname:
    #     shutil.move('project.pkl',projectname)

    return project, config


if __name__ == "__main__":
    # freeze_support()
    main()

from __future__ import annotations

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector

import numpy as np

def ensure_r_packages_installed():
    utils = importr('utils')
    base = importr('base')
    utils.chooseCRANmirror(ind=1)  # Select a CRAN mirror

    required_r_packages = ["BRISC"]
    for pkg in required_r_packages:
        if not base.require(StrVector([pkg]))[0]:
            print(f"Installing R package: {pkg}")
            utils.install_packages(StrVector([pkg]))
        else:
            print(f"R package: {pkg} installed")

# Ensure R packages are installed at runtime
ensure_r_packages_installed()

BRISC = importr('BRISC')

def BRISC_estimation(residual, X, coord):
    residual_r = ro.FloatVector(residual)
    coord_r = ro.FloatVector(coord.transpose().reshape(-1))
    coord_r = ro.r['matrix'](coord_r, ncol=2)

    if X is None:
        res = BRISC.BRISC_estimation(coord_r, residual_r)
    else:
        Xr = ro.FloatVector(X.transpose().reshape(-1))
        Xr = ro.r['matrix'](Xr, ncol=X.shape[1])
        res = BRISC.BRISC_estimation(coord_r, residual_r, Xr)

    theta_hat = res[9]
    beta = res[8]
    beta = np.array(beta)
    theta_hat = np.array(theta_hat)
    phi = theta_hat[2]
    tau_sq = theta_hat[1]
    sigma_sq = theta_hat[0]
    theta_hat[1] = phi
    theta_hat[2] = tau_sq / sigma_sq

    return beta, theta_hat







        

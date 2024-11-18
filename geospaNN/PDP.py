import warnings

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
from typing import Callable, Optional

import torch

class _PDP_estimator(BaseEstimator, RegressorMixin):
    def __init__(self, intValue=0):
        self.intValue = intValue
    def fit(self, X, model):
        self.treshold_ = 1
        self.model = model
        return self
    def _meaning(self, x):
        return 1
    def predict(self, X):
        return self.model.estimate(torch.from_numpy(X)).reshape(-1).detach().numpy()

def plot_PDP(model,
             X: torch.tensor,
             names: Optional[list] = [],
             save: bool = False):
    """Partial dependency plot for model on the data.

    A Partial Dependence Plot (PDP) is a visualization tool used to illustrate the relationship between a selected feature
    and the predicted outcome of a machine learning model, while averaging out the effects of other features.
    This helps to understand the marginal influence of a single feature on the model's predictions in a more interpretable way.

    Parameters:
        model:
            Usually a model in nngls class. Can take any model having a .estimate() method that take tensor X as input and
            predicted scalar value Y as output. (to implement for more models)
        X:
            nxp array of the covariates.
        names:
            List of names for variable, if not specified, use "variable 1" to "variable p".
        save:
            Whether to save the PDPs to the working directory. Default False.

    Returns:
        PDPs for each variable.

    See Also:
    Datta, Abhirup, et al. "Hierarchical nearest-neighbor Gaussian process models for large geostatistical datasets."
    Journal of the American Statistical Association 111.514 (2016): 800-812. \
    Katzfuss, Matthias & Guinness, Joseph. "A General Framework for Vecchia Approximations of Gaussian Processes."
    Statist. Sci. 36 (1) 124 - 141, February 2021. https://doi.org/10.1214/19-STS755
    """
    X = X.detach().numpy()
    p = X.shape[1]
    Est = _PDP_estimator()
    Est.fit(X, model)
    if len(names) != p:
        warnings.WarningMessage("length of names does not match columns of X, replace by variable index")
        names = [f"variable {i + 1}" for i in range(p)]

    for k in range(p):
        res = PartialDependenceDisplay.from_estimator(estimator = Est, X = X, features = [k],
                                                      feature_names = names,
                                                      percentiles=(0.05,0.95))
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,
                            wspace=0.4, hspace=0.4)
        if save:
            plt.savefig("./" + names[k] + ".png")

    return res
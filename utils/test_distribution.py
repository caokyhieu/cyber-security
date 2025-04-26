from scipy.stats import ks_2samp
import numpy as np
def is_two_distribution_different(x,y,p_thrshold=0.05):
    """
    Test two distributions by Two-sample Kolmogorov-Smirnov test
    """
    x = np.array(x).squeeze()
    y = np.array(y).squeeze()
    if x.ndim == 1:

        return ks_2samp(x, y).pvalue < p_thrshold
    elif x.ndim == 2:
        result = []
        for i in range(x.shape[-1]):
            result.append(ks_2samp(x[:,i], y[:,i]).pvalue)
        return np.mean(result) < p_thrshold


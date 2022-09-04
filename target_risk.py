######################################################################################################

import pandas as pd
import numpy as np
import scipy.optimize as so

######################################################################################################

def target_risk_weights_contributions(target_risk, cov_estimator, **kwargs):
    """
    
    Returns the weights that the contributions to portfolio risk are as close as possible
    to the target_risk, given the covariance matrix:

    1- There are three functions to calculate the covariance matrix, select one of them in
       cov_estimator.

    2- There are three functions to calculate the weights of portfolio, select one of them
       in weights_estimator.

    3- Three functions for covariance matrix:

            "sample_cov": Returns the sample covariance of the supplied returns.
            "constant_cor_cov": Estimates a covariance matrix by using the Elton/Gruber
                                constant correlation model.
            "shrinkage_cov": Covariance estimator that shrinks between the sample covariance
                             and the constant correlation estimators.

    4- Three functions to calculate the weights of portfolio:

            "weight_ew": Returns the weights of the equal weighted portfolio based on
                         the asset returns.
            "weight_cw": Returns the weights of the cap weighted portfolio based on
                         the time series of capweights.
            "weight_gmv": Produces the weights of the Global Minimum Volatility portfolio
                          given a covariance matrix of the returns.
    
    5- The "portfolio_return" function computes the return on a portfolio from constituent
       returns and weights.

    6- The "portfolio_vol" function computes the volatility of a portfolio from a covariance
       matrix and constituent weights.

    7- The "neg_sharpe" function is a part of "w_max_sharp_ratio" function and returns the
       negative of the sharpe ratio of the given portfolio.

    8- The "w_max_sharp_ratio" function returns the weights of the portfolio that gives you
       the maximum sharpe ratio given the riskfree rate, expected returns and covariance matrix.

    9- The "risk_contribution" function Computes the contributions to risk of the constituents
       of a portfolio, given a set of portfolio weights and a covariance matrix.

    10- The "mean_sd_risk" function returns the Mean Squared Difference in risk contributions
        between weights and target_risk(output of the "risk_contribution" function).

    """

######################################################################################################

    def portfolio_return(returns, weights_estimator = weight_ew, **kwargs):
        w = weights_estimator
        return w.T @ returns
    
    def portfolio_vol(weights_estimator = weight_ew, cov_estimator = sample_cov, **kwargs):
        w = weights_estimator
        cov = cov_estimator
        vol = (w.T @ cov @ w) ** 0.5
        return vol
    
    def neg_sharpe(riskfree_rate, expected_returns, weights_estimator = weight_ew, 
                   cov_estimator = sample_cov, **kwargs):
        cov = cov_estimator
        w = weights_estimator
        r = portfolio_return(w, expected_returns)
        vol = portfolio_vol(w, cov)
        return -(r - riskfree_rate) / vol
    
    def w_max_sharp_ratio(riskfree_rate, expected_returns, cov_estimator = sample_cov, **kwargs):
        cov = cov_estimator
        n = expected_returns.shape[0]
        init_guess = np.repeat(1 / n, n)
        bounds = ((0.0, 1.0),) * n 
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1}
        weights = so.minimize(neg_sharpe, init_guess,
                        args=(riskfree_rate, expected_returns, cov), method='SLSQP',
                        options={'disp': False},
                        constraints=(weights_sum_to_1,),
                        bounds=bounds)
        return weights.x
    
    def risk_contribution(weights_estimator = weight_ew, cov_estimator = sample_cov, **kwargs):
        w = weights_estimator
        cov = cov_estimator
        total_portfolio_var = portfolio_vol(w, cov) ** 2
        marginal_contrib = cov @ w
        risk_contrib = np.multiply(marginal_contrib, w.T) / total_portfolio_var
        return risk_contrib
    
    def mean_sd_risk(target_risk, weights_estimator = weight_ew, cov_estimator = sample_cov, **kwargs):
        w = weights_estimator
        cov = cov_estimator
        w_contribs = risk_contribution(w, cov)
        return ((w_contribs - target_risk) ** 2).sum()

                                ##############################################   
    
    def sample_cov(returns, **kwargs):
        return returns.cov()

    def constant_cor_cov(returns, **kwargs):
        rhos = returns.corr()
        n = rhos.shape[0]
        rho_bar = (rhos.values.sum() - n) / (n * (n-1))
        ccor = np.full_like(rhos, rho_bar)
        np.fill_diagonal(ccor, 1.)
        sd = returns.std()
        return pd.DataFrame(ccor * np.outer(sd, sd), index = returns.columns, columns = returns.columns)

    def shrinkage_cov(returns, delta, **kwargs):
        prior = constant_cor_cov(returns, **kwargs)
        sample = sample_cov(returns, **kwargs)
        return delta * prior + (1 - delta) * sample   

                                ##############################################
    
    def weight_ew(returns, cap_weights = None, max_cap_weights_multiple = None, 
                  microcap_threshold_percentage = None, **kwargs):
        n = len(returns.columns)
        ew = pd.Series(1/n, index=returns.columns)
        if cap_weights is not None:
            cw = cap_weights.loc[returns.index[0]] 

            if microcap_threshold_percentage is not None and microcap_threshold_percentage > 0:
                microcap = cw < microcap_threshold_percentage
                ew[microcap] = 0
                ew = ew / ew.sum()
            
            if max_cap_weights_multiple is not None and max_cap_weights_multiple > 0:
                ew = np.minimum(ew, cw * max_cap_weights_multiple)
                ew = ew / ew.sum()
        return ew

    def weight_cw(returns, cap_weights, **kwargs):
        w = cap_weights.loc[returns.index[1]]
        return w / w.sum()

    def weight_gmv(returns, **kwargs):
        cov = shrinkage_cov(returns, **kwargs)
        n = cov.shape[0]
        return w_max_sharp_ratio(0, np.repeat(1, n), cov)
    
######################################################################################################

    cov = cov_estimator
    n = cov.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n 
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1}
    weights = so.minimize(mean_sd_risk, init_guess,
                       args=(target_risk, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x
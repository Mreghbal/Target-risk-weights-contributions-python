# Target-risk-weights-contributions-python
Returns the weights that the contributions to portfolio risk are as close as possible to the target risk, given the covariance matrix

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

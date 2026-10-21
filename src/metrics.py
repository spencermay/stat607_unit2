"""
Calculate the MSE for the bayes rule. I'm no Empirical Bayesian,
so maybe this is not the standard metric, but it works, no?
"""

def mse_bayes_rule(estimate, oracle, true_dgp):
    """
    E_{ybar,S \\sim DGP} [(estimate(ybar, S)-oracle(ybar, S))^2]
    """
    
# Stat 607 Unit 2 Project: Simulation Study Design

Implement Figure 4 from the paper
> Unobserved Heterogeneity in Income Dynamics: An Empirical Bayes Perspective by Jiaying Gu, Roger Koenker

Suppose we observe data
$$y_{it}=\alpha_i+\sqrt{\theta_i}u_{it}, \quad u_{it}\sim N(0,1)$$
drawn from normal distributions with their location and scale $\alpha,\theta$ drawn from some hidden distribution. This paper uses the NPMLE estimator to match the distribution of observed values and then infer a Bayes rule from that (decision rule given ybar and the ith sample's std deviation).

Run all with ```make all```. Run tests with ```make test PYTEST_ARGS='-m "not slow"'``` (took about 12s on 16Gb Mac; full version takes 42s, which is just ```make test```). Clean the workspace with ```make clean```.

### Key results:
As I record in the table below from metrics.py (that Claude so generously formatted for me), the NPMLE recovers the Bayes rule in both the independent $(\alpha, \theta)$ and correlated settings, according to the regret (risk minus oracle risk) of each. Compared to the no-shrinkage method (Naive in the table below), the npmle has orders of magnitude lower regret (when we consider the NPMLE method that matches the correlated/independent setup; even so, the other method still has vastly preferable regret to the naive method). Note, this might be a bit of a strawman though because no shrinkage is clearly a bad choice---is there another EB approach, I don't know. 

## Table from metrics.py (```make analyze```):

-------
           Monte Carlo risk of Bayes-rule estimators for the mean (alpha)           
         Gaussian location-scale mixture;  n = 400 units,  m = 11 obs/unit          
-------
Two risk measures are: 

- Regret vs oracle : MSE of the estimator's rule against the oracle rule
    - E[alpha | ybar, S]  -- excess risk from not knowing H.

- Total vs alpha   : MSE against the true latent alpha = total estimation risk.

Lower is better; the oracle attains zero regret by definition.

### Setting A: alpha independent of theta

| Estimator | Regret vs oracle | Total vs alpha |
| ---- | ----- | ----- |
| Naive (no shrinkage, alpha = ybar) | 0.1043 ± 0.0013 | 0.1979 ± 0.0022 |
| Oracle (knows true H)  | 0.0000 ± 0.0000 | 0.0935 ± 0.0021 |
| NPMLE (independence assumed) | 0.0040 ± 0.0007 | 0.0979 ± 0.0077 |
| NPMLE (joint alpha, theta) | 0.0131 ± 0.0024 | 0.1068 ± 0.0078 |


Setting B: alpha positively correlated with theta
------------------------------------------------------------------------------------
| Estimator |                            Regret vs oracle       | Total vs alpha |       
| --------------| ---------------------------| ------------------------------------------- |
| Naive (no shrinkage, alpha = ybar)  |  0.1338 ± 0.0013       |  0.1976 ± 0.0022       |
|Oracle (knows true H)        |         0.0000 ± 0.0000     |    0.0628 ± 0.0020       |
|NPMLE (independence assumed)    |      0.0400 ± 0.0029     |    0.1088 ± 0.0075       |
|NPMLE (joint alpha, theta)   |         0.0087 ± 0.0013    |     0.0774 ± 0.0080       |

------------------------------------------------------------------------------------
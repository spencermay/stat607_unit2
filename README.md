# Stat 607 Unit 2 Project: Simulation Study Design

Implement Figure 4 from the paper
> Unobserved Heterogeneity in Income Dynamics: An Empirical Bayes Perspective by Jiaying Gu, Roger Koenker

Suppose we observe data
$$y_{it}=\alpha_i+\sqrt{\theta_i}u_{it}, \quad u_{it}\sim N(0,1)$$
drawn from normal distributions with their location and scale $\alpha,\theta$ drawn from some hidden distribution. This paper uses the NPMLE estimator to match the distribution of observed values and then infer a Bayes rule from that (decision rule given ybar and the ith sample's std deviation).
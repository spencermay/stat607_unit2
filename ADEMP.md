# ADEMP
| Component | Description |
| ------------------------------ | ----------- |
| **A — Aims** | - In the Gaussian location-scale mixture model, how well does the NPMLE Emprical Bayes method recover the distribution of the mean and variance? |
| | - How well does this model recover the density of y? |
| | - How well does this recover the Bayes rule? |
| **D — DGMs** | Just to have the model here for comparison, we observe $ y_i = \alpha_i + \sqrt{\theta_i} u_{it} $, with $ u_{it}\sim N(0,1) $. |
| | - $n=400, m=11$ (same as paper). |
| | - We must specify the distribution of the mean and variance: iid $ \alpha_i \sim F $, and iid $\theta_i \sim G$ (suppose $\alpha, \theta$ are independent). At a minimum, I will implement the case: $ \alpha_{i} \sim\frac{1}{3}\delta_{-0.5}+{\frac{1}{3}}\delta_{1}+{\frac{1}{3}} \delta_{3}, $ and iid $ \theta_{i} \sim\frac{1}{3}\delta_{0.5}+{\frac{1}{3}}\delta_{2}+{\frac{1}{3}} \delta_{4}, $ because this is the setup they use in Figure 4 which I think encapsulates the Aims well.
| **E — Estimands (or targets)** | - Mixing distributions for mean ($\alpha$) and variance ($\theta$)|
|| - Bayes rule for alpha |
| **M — Methods** | NPMLE and EM algorithm -- n_grid = 300 (number of gridpoints in each npmle estimator for the pmfs of mean, variance) |
|| - compare to naive bayes rule: $\hat{\alpha}=\bar{x}$ |
|| - compare to oracle bayes rule |
| **P — Performance measures** | - MSE of Bayes rule (under the true DGP) |
|| - shape comparison (qualitative -- spaghetti plot for $n_{sim}=100$)|



----


## Simulation conditions: (minimum)

| $ \alpha_i \sim F $ |$\quad$ $\theta_i \sim G$|
|---|---|
| $ \alpha_{i} \sim\frac{1}{3}\delta_{-0.5}+{\frac{1}{3}}\delta_{1}+{\frac{1}{3}} \delta_{3}, $ |$\quad$ $ \theta_{i} \sim\frac{1}{3}\delta_{0.5}+{\frac{1}{3}}\delta_{2}+{\frac{1}{3}} \delta_{4}, $|
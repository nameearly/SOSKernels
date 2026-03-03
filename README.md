# SOS Kernels
Reproducing experiments from "Non-parametric Models for Non-negative Functions" ([arxiv](https://arxiv.org/abs/2007.03926)).

## Quick Start
This repo was tested with Julia 1.12.1. After cloning the repo, instantiate the environment.

In the shell:
```
julia --project
# press `]` to enter `(SOSKernels) pkg>` mode.
instantiate
```

This will install all required packages. An unfortunate dependency is [MOSEK](https://www.mosek.com/products/academic-licenses/); open source tools like Clarabel simply don't work for these problems. Follow the link to request a free academic license.

## Experiments
### Density Estimation
Consider iid calibration samples $x_1, ..., x_n$. We want to estimate the generating distribution $P(x)$. Using independence we can write $P(x) = P(x | x_1, ..., x_n)$. Use a sum-of-squares kernel model for this conditional probability. That is, for $B\succeq 0$:

$$
P(x | x_1, ..., x_n) = f_B(x) = v(x)^T B v(x),
$$

where $v(x) = [k(x,x_1), ..., k(x,x_n)]$ and $k$ is our kernel function (Gaussian kernel, in this case).

The maximum likelihood estimator (with regularization) is:

$$
\max_B \prod_{i=1}^nf_B(x_i) + \Omega(B)\quad\mathrm{s.t.}\int_\mathcal{X} f_B(x) = 1.
$$

We can easily model the integral constraint using our SOS model. To handle the product in a tractable way, we take the log likelihood:

$$
\max_B \sum_{i=1}^n\log(f_B(x_i)) + \Omega(B)\quad\mathrm{s.t.}\int_\mathcal{X} f_B(x) = 1.
$$

To run from the Julia REPL:
```julia
include("experiments/density_estimation.jl")
```

The model is, in general, quite sensitive to the choice of kernel (and regularization) parameters. For example:

  $\sigma = 1$  |  $\sigma = 0.6$  |  $\sigma = 0.3$
:-------------------------:|:-------------------------:|:-:
![](results/density_estimation_1.png) | ![](results/density_estimation_6.png) | ![](results/density_estimation_3.png)

Note that I had to manually tune the regularization term. The $\sigma = 1$ case (from the paper) really isn't so great; the smoothness priors override the function. $\sigma=0.6$ seems to be a sweet spot, while decreasing $\sigma$ any smaller leads to big oscillations. This is a frustrating part of kernel estimation: it finds an "optimal" representation that is highly sensative to the choice of parameters.

Opportunity: a better way to set kernel parameters.

**TODO: compare with classical density estimation (Glivenko-Cantelli)**

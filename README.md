# Probabilistic Inference

Exact and approximate inference in probabilistic models: independence in
graphical models, an exact likelihood on a tree by dynamic programming, then
two approximations — mean-field variational inference and EM.

Assignment 2 of **DD2434 Machine Learning, Advanced Course**, KTH, autumn 2021.

📄 **[Read the report](report.pdf)** · [assignment brief](assignment.pdf)

## 2.1 — Graphical models

Six d-separation questions, answered in the report: which variables are
independent of which, given what.

## 2.2 — [Exact likelihood on a tree](tree-likelihood)

Given a rooted binary tree $T$, categorical CPDs $\theta$, and an assignment
$\beta$ of values to the leaves, compute $p(\beta \mid T, \theta)$.

The sum over all assignments to the internal nodes is exponential, so
[`likelihood.py`](tree-likelihood/likelihood.py) does it by recursion instead.
Define $s(v, i)$ as the probability of everything observed below $v$ given
$X_v = i$; for a node with children $u, w$,

$$s(v,i) = \left(\sum_j p(X_u = j \mid X_v = i)\, s(u,j)\right)\left(\sum_j p(X_w = j \mid X_v = i)\, s(w,j)\right)$$

with $s(l,i) = \mathbb{1}[x_l = i]$ at the leaves, and the answer read off at the
root as $\sum_i s(r,i)\, p(X_r = i)$. One bottom-up pass, linear in the number of
nodes.

Runs on three trees of increasing size in [`data/`](tree-likelihood/data):

```bash
cd tree-likelihood && python likelihood.py
```

## 2.3 — [Variational inference](variational-inference)

A univariate Gaussian with both mean and precision unknown, given a
Normal–Gamma prior. The true posterior does not factorise, but the mean-field
assumption $q(\mu, \tau) = q(\mu)\,q(\tau)$ makes the updates closed-form:
$q(\mu)$ stays Normal, $q(\tau)$ stays Gamma, and the two are iterated to
convergence.

[`vi_gaussian.py`](variational-inference/vi_gaussian.py) runs those updates and
plots the approximation against the exact posterior, so the cost of the
factorisation assumption is visible — the approximation is correctly centred but
too narrow, because independence cannot represent the correlation between $\mu$
and $\tau$. [`figures/`](variational-inference/figures) shows how the fit changes
with sample size and with the prior.

## 2.4 — [EM for mixture models](em-mixture-models)

EM written from scratch and checked against scikit-learn:

| | from scratch | scikit-learn |
|---|---|---|
| the assignment's data | [`gmm_from_scratch_data.py`](em-mixture-models/gmm_from_scratch_data.py) | [`gmm_sklearn_data.py`](em-mixture-models/gmm_sklearn_data.py) |
| a worked example | [`gmm_from_scratch_example.py`](em-mixture-models/gmm_from_scratch_example.py) | [`gmm_sklearn_example.py`](em-mixture-models/gmm_sklearn_example.py) |

[`em_gaussian_poisson.py`](em-mixture-models/em_gaussian_poisson.py) extends it
to a mixture with both Gaussian and Poisson components.
[`figures/`](em-mixture-models/figures) shows the fit for one to five clusters.

Singular covariances are avoided by adding a small multiple of the identity at
each M-step.

```bash
pip install numpy scipy matplotlib scikit-learn
```

import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(314)
N = 100
alpha_real = 2.5
beta_real = [0.9, 1.5]
eps_real = np.random.normal(0, 0.5, size=N)

X = np.array([np.random.normal(i, j, N) for i, j in zip([10, 2], [1, 1.5])])
X_mean = X.mean(axis=1, keepdims=True)
X_centered = X - X_mean
y = alpha_real + np.dot(beta_real, X) + eps_real


# 多元回归

def scatter_plot(x, y):
    plt.figure(figsize=(10, 10))
    for idx, x_i in enumerate(x):
        plt.subplot(2, 2, idx + 1)
        plt.scatter(x_i, y)
        plt.xlabel('$x_{}$'.format(idx), fontsize=16)
        plt.ylabel('$y$', fontsize=16)

    plt.subplot(2, 2, idx + 2)
    plt.scatter(X[0], X[1])
    plt.xlabel('$x_{}$'.format(idx - 1), fontsize=16)
    plt.ylabel('$y_{}$'.format(idx), fontsize=16)
    plt.show()

scatter_plot(X_centered, y)
if __name__ == '__main__':
    with pm.Model() as model_mlr:
        alpha_tmp = pm.Normal('alpha_tmp', mu=0, sd=10)
        beta = pm.Normal('beta', mu=0, sd=1, shape=2)
        epsilon = pm.HalfCauchy('epsilon', 5)

        mu = alpha_tmp+pm.math.dot(beta, X_centered)

        alpha = pm.Deterministic('alpha', alpha_tmp - pm.math.dot(beta, X_mean))

        y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y)

        start = pm.find_MAP()
        step = pm.NUTS(scaling=start)
        trace_mlr = pm.sample(5000, step=step, start=start)
varsname = ['alpha', 'beta', 'epsilon']
pm.traceplot(trace_mlr, varsname)
plt.show()
data = pm.df_summary(trace_mlr, varsname)
print(data)





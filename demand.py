import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt


# Set-up.
n = 10000
# Parameters of the mixture components
norm_params = np.array([[6, 3],
                        [16, 5]])
n_components = norm_params.shape[0]
# Weight of each component, in this case all of them are 1/3
weights = np.ones(n_components, dtype=np.float64) / 2.0
# A stream of indices from which to choose the component
mixture_idx = np.random.choice(len(weights), size=n, replace=True, p=weights)
# y is the mixture sample
y = np.fromiter((ss.norm.rvs(*(norm_params[i])) for i in mixture_idx),
                   dtype=np.float64)

# Theoretical PDF plotting -- generate the x and y plotting positions
xs = np.linspace(y.min(), y.max(), 200)
xs = np.linspace(y.min(), y.max(), 24)
ys = np.zeros_like(xs)

for (l, s), w in zip(norm_params, weights):
    ys += ss.norm.pdf(xs, loc=l, scale=s) * w

plt.scatter(xs, ys*75)
# plt.hist(y, normed=True, bins="fd")
plt.xlabel("hour")
plt.ylabel("average demand")
plt.xlim(0,24)
# plt.plot(hour_mus)
plt.show()

# import math
# mu = 3
# variance = 1
# sigma = math.sqrt(variance)
# x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
# plt.plot(x, ss.norm.pdf(x, mu, sigma))
# plt.xlabel("value")
# plt.ylabel("probability")
# plt.xlim(0)
# plt.show()

# x = np.arange(0, 100, 1)
# y = ss.poisson.pmf(x, mu=5, loc=0)
# plt.plot(x,y)
# plt.xlim(0,24)
# plt.xlabel("demand")
# plt.ylabel("prob")


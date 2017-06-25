import numpy as np
import matplotlib.pyplot as plt
import pymc as pm

N = 100
X = 0.025 * np.random.randn(N)
Y = 0.5 * X + 0.01 * np.random.randn(N)

ls_coeff = np.cov(X, Y)[0,1]/np.var(X)
ls_intercept = Y.mean() - ls_coeff * X.mean()

print "least squares coeff = %s" % ls_coeff
print "least squares intercept = %s" % ls_intercept


plt.scatter(X,Y, c="k")
plt.xlabel("Trading Signal")
plt.ylabel("Returns")
plt.title("Empirical Returns")
plt.plot(X,ls_coeff*X + ls_intercept, label="least squares line")
plt.xlim(X.min(),X.max())
plt.ylim(Y.min(),Y.max())
plt.legend(loc="upper left")



std = pm.Uniform("std", 0, 100, trace=False)

@pm.deterministic
def prec(U=std):
    return 1.0 / U **2

beta = pm.Normal("beta", 0, 0.0001)
alpha = pm.Normal("alpha", 0, 0.0001)

@pm.deterministic
def mean(X=X, alpha=alpha, beta=beta):
    return alpha + beta * X

obs = pm.Normal("obs", mean, prec, value=Y, observed=True)

mcmc = pm.MCMC([obs, beta, alpha, std, prec])
mymap = pm.MAP([obs, beta, alpha, std, prec])
mymap.fit()



mcmc.sample(100000, 80000)
pm.Matplot.plot(mcmc)

for s in mymap.stochastics:
    print "%s = %s" % (s, s.value)


    
    









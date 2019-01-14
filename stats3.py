from __future__ import print_function
import statsmodels.api as sm
import pandas
from patsy import dmatrices

df = sm.datasets.get_rdataset("Guerry", "HistData").data

vars = ['Department', 'Lottery', 'Literacy', 'Wealth', 'Region']
df = df[vars]
print(df[-5:])

df = df.dropna()
print(df[-5:])

y, X = dmatrices('Lottery ~ Literacy + Wealth + Region', data=df, return_type='dataframe')
print(y[:3])
print(X[:3])

mod = sm.OLS(y, X)    # Describe model
res = mod.fit()       # Fit model
print(res.summary())   # Summarize model

print(res.params)
print(res.rsquared)

print(sm.stats.linear_rainbow(res))

axes = sm.graphics.plot_partregress('Lottery', 'Wealth', ['Region', 'Literacy'], data=df, obs_labels=False)
print(axes)

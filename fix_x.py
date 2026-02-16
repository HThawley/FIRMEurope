import pandas as pd
import numpy as np

xpd = pd.read_csv("C:/Users/u6942852/Downloads/x.csv", header=None)
x = xpd.to_numpy().flatten()

gen = pd.read_csv("C:/Users/u6942852/Downloads/generators.csv")
res = pd.read_csv("C:/Users/u6942852/Downloads/reservoirs.csv")
sto = pd.read_csv("C:/Users/u6942852/Downloads/storages.csv")

lin = pd.read_csv("inputs/config/lines.csv")

gen_new = pd.read_csv("inputs/config/generators.csv")
res_new = pd.read_csv("inputs/config/reservoirs.csv")
sto_new = pd.read_csv("inputs/config/storages.csv")

lgen = len(gen)
lres = len(res)
lsto = len(sto)

llin = len(lin) - 6

lgen_n = len(gen_new)
lres_n = len(res_new)
lsto_n = len(sto_new)

x_new = -1 * np.ones(lgen_n + 2*lres_n + 2*lsto_n + llin, float)

i, j = 0, 0
while i < lgen:
    if gen.loc[i, "name"] in gen_new["name"].values:
        x_new[j] = x[i]
        j += 1
    i += 1

i, j = 0, 0
while i < lres:
    if res.loc[i, "name"] in res_new["name"].values:
        x_new[lgen_n+j] = x[lgen+i]
        j += 1
    i += 1

i, j = 0, 0
while i < lres:
    if res.loc[i, "name"] in res_new["name"].values:
        x_new[lgen_n+lres_n+j] = x[lgen+lres+i]
        j += 1
    i += 1

i, j = 0, 0
while i < lsto:
    if sto.loc[i, "name"] in sto_new["name"].values:
        x_new[lgen_n+lres_n*2+j] = x[lgen+lres*2+i]
        j += 1
    i += 1

i, j = 0, 0
while i < lsto:
    if sto.loc[i, "name"] in sto_new["name"].values:
        x_new[lgen_n+lres_n*2+lsto_n+j] = x[lgen+lres*2+lsto+i]
        j += 1
    i += 1

i, j = 0, 0
while i < llin:
    x_new[lgen_n+lres_n*2+lsto_n*2+j] = x[lgen+lres*2+lsto*2+i]
    j += 1
    i += 1

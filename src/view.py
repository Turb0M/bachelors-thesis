import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from preprocessing import prepare

def view(file, denom, title=""):
    rng = np.random.default_rng()
    data = prepare(file, rng, denominator=denom)

    plt.scatter(data[:,0], data[:,1], marker='.', s=2.0)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig("view.png", dpi=300)


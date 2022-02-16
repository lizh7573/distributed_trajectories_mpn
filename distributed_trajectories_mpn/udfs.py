"""
User Defined Functions
======================
"""


import os
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import pyspark.sql.functions as F
from random import seed, random
from pyspark.ml.linalg import Vectors
from pyspark.sql import Window
from pyspark.sql.types import IntegerType




def matrix_updates(states1, states2):
    update = [[float(el1[0]), float(el2[0]), el1[1]*el2[1]]
             for el1 in states1 for el2 in states2]
    return update


def prepare_for_plot(data, type_):

    pd_df = data.toPandas()

    data = np.array( pd_df[type_] )
    rows = np.array( pd_df['y'].astype('int') )
    cols = np.array( pd_df['x'].astype('int') )

    M = sparse.coo_matrix((data, (rows, cols)), shape = (4069+1, 4069+1))

    return M


def plot_sparse(matrix, fname, title, dirname):
    plt.figure(figsize = (20, 20))
    plt.spy(matrix, markersize = 10, alpha = 0.5)
    plt.grid()
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.xlabel("polygon", fontsize = 27)
    plt.ylabel("polygon", fontsize = 27)
    plt.title(title, fontsize = 30)
    plt.savefig(os.path.join(dirname, fname))



def plot_dense(matrix, fname, title, dirname):
    plt.figure(figsize = (20, 20))
    plt.imshow(matrix.todense())
    plt.colorbar()
    plt.grid()
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.xlabel("polygon", fontsize = 27)
    plt.ylabel("polygon", fontsize = 27)
    plt.title(title, fontsize = 30)
    plt.savefig(os.path.join(dirname, fname))
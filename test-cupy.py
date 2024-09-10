# %%
# Import necessary libraries
import numpy as np
import pandas as pd
import cupy as cp
import cudf
from timeit import timeit

# Create a large random dataset
n = 10**9
np_data = np.random.rand(n)
cp_data = cp.random.rand(n)

# Create a pandas DataFrame and a cudf DataFrame
pd_df = pd.DataFrame({"data": np_data})
cudf_df = cudf.DataFrame({"data": cp_data})

# Define functions to perform operations


def numpy_sum():
    return np.sum(np_data)


def pandas_sum():
    return pd_df["data"].sum()


def cupy_sum():
    return cp.sum(cp_data)


def cudf_sum():
    return cudf_df["data"].sum()


# Time the operations
numpy_time = timeit(numpy_sum, number=10)
pandas_time = timeit(pandas_sum, number=10)
cupy_time = timeit(cupy_sum, number=10)
cudf_time = timeit(cudf_sum, number=10)

# Print the results
print("Numpy sum time:", numpy_time)
print("Pandas sum time:", pandas_time)
print("Cupy sum time:", cupy_time)
print("Cudf sum time:", cudf_time)

# Import necessary libraries
import numpy as np
import pandas as pd
from timeit import timeit

# Create a large random dataset
n = 10**9
np_data = np.random.rand(n)

# Create a pandas DataFrame
pd_df = pd.DataFrame({"data": np_data})

# Define functions to perform operations


def numpy_sum():
    return np.sum(np_data)


def pandas_sum():
    return pd_df["data"].sum()


# Time the operations
numpy_time = timeit(numpy_sum, number=10)
pandas_time = timeit(pandas_sum, number=10)

# Print the results
print("Numpy sum time:", numpy_time)
print("Pandas sum time:", pandas_time)

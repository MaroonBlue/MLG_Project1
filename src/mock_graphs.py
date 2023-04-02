import numpy as np
from pandas import DataFrame

from src.graph import *


def get_mock_wheel_graph(n: int):
    return WheelGraph(n)

def get_mock_fully_connected_graph(n: int):
    df = DataFrame(
        {
            "from": [i for i in range(n) for x in range(n - 1)],
            "to": [x for i in range(n) for x in range(n) if x != i]
        }
    )
    return DataFrameGraph(df)

def get_mock_dataframe_graph(n: int):
    df = DataFrame(
        {
            "from": [i for i in range(n) for x in range(n - 1)],
            "to": [x for i in range(n) for x in range(n) if x != i]
        }
    )
    return DataFrameGraph(df)

def get_mock_random_graph(n: int, n_edges: int = None, verify = True):
    if n_edges is None:
        n_edges = n * n // 2 
    src = [i for i in range(n) for x in range(n - 1)]
    dst = [x for i in range(n) for x in range(n) if x != i]
    all_pairs = np.array(list(zip(src, dst)))
    all_pairs_size = all_pairs.shape[0]

    for _ in range(1000): # After 1000 failed attempts stop generating
        try:
            selected_pair_ids = np.random.choice(range(all_pairs_size), n_edges, replace=False)
            selected_pairs = all_pairs[selected_pair_ids, :]
            df = DataFrame(
                {
                    "from": selected_pairs[:,0],
                    "to": selected_pairs[:,1]
                }
            )
            graph = DataFrameGraph(df, verify_connected=verify)
            return graph
        except Exception as e:
            print(e)
            


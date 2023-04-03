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
    return get_mock_fully_connected_graph(n)

def get_mock_random_graph(n: int, n_edges: int = None, n_threshold = None):
    if n_edges is None:
        n_edges = n * n // 2 
    if n_threshold is None:
        n_threshold = int(0.8*n)

    src = [i for i in range(n) for x in range(n - 1)]
    dst = [x for i in range(n) for x in range(n) if x != i]
    all_pairs = np.array(list(zip(src, dst)))
    all_pairs_size = all_pairs.shape[0]

    graph = None
    for _ in range(1000): # After 10 failed attempts stop generating
        try:
            selected_pair_ids = np.random.choice(range(all_pairs_size), n_edges, replace=False)
            selected_pairs = all_pairs[selected_pair_ids, :]
            l = list(set(selected_pairs[:,0].tolist()))
            if len(l) < n_threshold: raise Exception()
            df = DataFrame(
                {
                    "from": selected_pairs[:,0],
                    "to": selected_pairs[:,1]
                }
            )
            graph = DataFrameGraph(df)
            return graph
        except:
            pass
    raise Exception(f"Failed to generate random graph with at least {n_threshold} nodes and {n_edges} edges")
            


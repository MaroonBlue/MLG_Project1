from pandas import DataFrame

from src.graph import *


def get_mock_wheel_graph(n: int):
    return WheelGraph(n)

def get_mock_dataframe_graph(n: int):
    df = DataFrame(
        {
            "from": [i for i in range(n) for x in range(n - 1)],
            "to": [x for i in range(n) for x in range(n) if x != i]
        }
    )
    return DataFrameGraph(df)


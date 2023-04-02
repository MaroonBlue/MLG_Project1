from src.graph import WheelGraph, NumpyGraph, DataFrameGraph
from src.fastgraph import FastGraph, FastGraphSettings
from src.mock_graphs import get_mock_wheel_graph, get_mock_dataframe_graph

graph = get_mock_wheel_graph(50)
settings = FastGraphSettings(
    subgraph_size = 5,
    render = True,
    render_isomorphic_graphs = True,
    render_x_isomorphisms_per_column = 4
)

FastGraph(
    graph,
    settings
)


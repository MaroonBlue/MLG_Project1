from src.graph import WheelGraph, NumpyGraph, DataFrameGraph
from src.fastgraph import FastGraph, FastGraphSettings
from src.mock_graphs import get_mock_wheel_graph, get_mock_dataframe_graph

graph = get_mock_wheel_graph(1000)
settings = FastGraphSettings(
    render = True, 
    render_isomorphic_graphs = True, 
    render_x_isomorphisms_per_column = 6, 
    subgraph_size = 3,
    letters_per_sentence = 25,
    sentences_to_generate = 1000,
    spacebar_probability = 0.1,
    end_of_sentence_symbol = '\n'
)

FastGraph(
    graph,
    settings
)


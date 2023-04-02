from src.graph import WheelGraph, NumpyGraph, DataFrameGraph
from src.fastgraph import FastGraph, FastGraphSettings
from src.mock_graphs import get_mock_wheel_graph, get_mock_fully_connected_graph, get_mock_random_graph

# graph = get_mock_fully_connected_graph(8)
graph = get_mock_random_graph(10, 10)

settings = FastGraphSettings(
    render = True, 
    render_isomorphic_graphs = True,
    render_auto_walk_delay_seconds = 0.0001,
    render_x_isomorphisms_per_column = 6, 
    subgraph_size = 4,
    letters_per_sentence = 25,
    sentences_to_generate = 1000,
    spacebar_probability = 0.1,
    end_of_sentence_symbol = '\n'
)

FastGraph(
    graph,
    settings
)


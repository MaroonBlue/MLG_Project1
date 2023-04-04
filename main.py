from src.mock_graph import get_mock_wheel_graph, get_mock_fully_connected_graph, get_mock_random_graph
from src.graph_from_data import get_facebook_dataframe_graph, get_deezer_dataframe_graph
from src.fastgraph import FastGraph, FastGraphSettings

# Use any of these graphs for testing

# graph = get_facebook_dataframe_graph()
# graph = get_mock_fully_connected_graph(8)
# graph = get_mock_wheel_graph(8)
graph = get_mock_random_graph(40, 60)
# graph = get_deezer_dataframe_graph(0)

settings = FastGraphSettings(
    render = True,                              # Display GUI - if disabled, automatically starts sentences generation
    render_isomorphic_graphs = True,            # Render isomorphisms on the side - disabling increases the max rendering fps
    render_auto_walk_delay_seconds = 0.0001,    # How often to update the subgraph. Note that this is a delay between each step, not FPS
    render_x_isomorphisms_per_column = 6,       # Display isomorphisms in columns of given amount of elements each, stylistic choice
    subgraph_size = 5,                          # Size of the isomorphic subgraphs to generate and walk with
    letters_per_sentence = 50,                  # Self explanatory
    sentences_to_generate = 1000,               # Self explanatory
    spacebar_probability = 0.1,                 # Chance of inserting spacebar after each symbol
    end_of_sentence_symbol = '\n',              # Symbol at the end of each sentence
    file_name = "facebook_dataframe_graph"      # Name of the file to use for the output. A .txt file extension will be added automatically
)

FastGraph(
    graph,
    settings
)


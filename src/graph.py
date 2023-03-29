import networkx as nx
import numpy as np

class Graph:
    def __init__(self, nx_graph: nx.graph, node_distance = 0.1, iterations = 10) -> None:
        graph_dict = nx.spring_layout(
            nx_graph, 
            k = node_distance, 
            iterations = iterations
        )
        
        nodes_count = len(graph_dict)
        node_names = [''] * nodes_count
        node_positions = np.zeros((nodes_count, 2))
        node_name_position_map = {}

        for i, node_name in enumerate(graph_dict.keys()):
            node_position = graph_dict[node_name]
            node_names[i] = node_name
            node_positions[i, :] = node_position
            node_name_position_map[node_name] = node_position

        self.nx_graph = nx_graph
        self.graph_dict = graph_dict
        self.node_names = node_names
        self.node_positions = node_positions
        self.node_name_position_map = node_name_position_map

    def render(self, axis, with_labels = True):

        nx.draw_networkx(
            self.nx_graph, 
            self.graph_dict, 
            ax = axis,
            font_size=6, 
            node_color='#A0CBE2', 
            edge_color='#BB0000', 
            width=0.2,
            node_size=20, 
            with_labels=with_labels,
        )

class WheelGraph(Graph):
    def __init__(self, nodes_count: int):
        nx_graph = nx.wheel_graph(nodes_count)
        Graph.__init__(self, nx_graph)

class DataFrameGraph(Graph):
    def __init__(self, dataframe, source_column_name = 'from', destination_column_name = 'to'):
        nx_graph = nx.from_pandas_edgelist(dataframe, source_column_name, destination_column_name)
        Graph.__init__(self, nx_graph)

class NumpyGraph(Graph):
    def __init__(self, array):
        nx_graph = nx.from_numpy_array(array)
        Graph.__init__(self, nx_graph)
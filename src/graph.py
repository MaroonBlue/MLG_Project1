import networkx as nx
import numpy as np
import matplotlib.axes as plt

class Graph:
    def __init__(self, nx_graph: nx.graph, node_distance = 0.1, iterations = 10, verify_connected = True) -> None:
        graph_dict = nx.spring_layout(
            nx_graph, 
            k = node_distance, 
            iterations = iterations
        )
        
        nodes_count = len(graph_dict)
        node_ids = [''] * nodes_count
        node_positions = np.zeros((nodes_count, 2))

        node_id_position_map = {}
        for i, node_id in enumerate(graph_dict.keys()):
            node_position = graph_dict[node_id]
            node_ids[i] = str(node_id)
            node_positions[i, :] = node_position
            node_id_position_map[str(node_id)] = node_position

        node_id_edges_map = {}
        for source_node_id, target_node_id in nx_graph.edges():
            source_node_id = str(source_node_id)
            target_node_id = str(target_node_id)

            if source_node_id in node_id_edges_map.keys():
                node_id_edges_map[source_node_id].add(target_node_id)
            else:
                node_id_edges_map[source_node_id] = set(target_node_id)

            if target_node_id in node_id_edges_map.keys():
                node_id_edges_map[target_node_id].add(source_node_id)
            else:
                node_id_edges_map[target_node_id] = set(source_node_id)

        if verify_connected:
            for node_id in node_ids:
                if node_id not in node_id_edges_map.keys():
                    raise AssertionError(f"Graph is not connected! Node without edges: {node_id}")

        self.nx_graph = nx_graph
        self.graph_dict = graph_dict
        self.axis: plt.Axes = None

        self.node_ids = node_ids
        self.node_id_edges_map = node_id_edges_map
        self.node_id_position_map = node_id_position_map

    def render(self, with_labels = True):
        if self.axis is None: return
        nx.draw_networkx(
            self.nx_graph, 
            self.graph_dict, 
            ax = self.axis,
            font_size=6, 
            node_color='#A0CBE2', 
            edge_color='#000000', 
            width=0.2,
            node_size=20, 
            with_labels=with_labels,
        )

    def highlight(self):
        self.axis.spines['bottom'].set(color = 'green', linewidth = 3)
        self.axis.spines['top'].set(color = 'green', linewidth = 3)
        self.axis.spines['left'].set(color = 'green', linewidth = 3)
        self.axis.spines['right'].set(color = 'green', linewidth = 3)
    
    def hide_border(self):
        self.axis.spines['bottom'].set(color = 'white', linewidth = 3)
        self.axis.spines['top'].set(color = 'white', linewidth = 3)
        self.axis.spines['left'].set(color = 'white', linewidth = 3)
        self.axis.spines['right'].set(color = 'white', linewidth = 3)
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
class IsomorphismGraph(Graph):
    def __init__(self, array, array2):
        self.isomorphisms = [array, array2]
        nx_graph = nx.from_numpy_array(array)
        Graph.__init__(self, nx_graph, verify_connected=False)

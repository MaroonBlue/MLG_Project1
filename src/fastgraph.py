import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec as GridSpec
from matplotlib.gridspec import GridSpecFromSubplotSpec as Grid

from numpy import zeros, array_equal
from threading import Thread, Event
from string import ascii_lowercase
from numpy.random import choice
from random import randint
from sys import stdout
from time import sleep

from src.graph import Graph
from src.isomorphic_graphs import get_all_unique_graphs, nauty_normalize

class FastGraphSettings:
    def __init__(self, 
                 render = False, 
                 render_isomorphic_graphs = False, 
                 render_x_isomorphisms_per_column = 6, 
                 subgraph_size = 3) -> None:

        self.render = render
        self.render_isomorphic_graphs = render_isomorphic_graphs
        self.render_x_isomorphisms_per_column = render_x_isomorphisms_per_column
        self.subgraph_size = subgraph_size

        def assert_that(condition, message):
            if not condition:
                raise AssertionError(message)

        assert_that(subgraph_size >= 2 and subgraph_size <= 6, "Subgraph size should be in range [2,6]")

class FastGraph:  
    def __init__(self, 
                 graph: Graph, 
                 settings: FastGraphSettings) -> None:

        self.graph = graph
        self.settings = settings
        self.prepare_subgraphs()

        if settings.render:
            self.prepare_interface()
            self.prepare_markers()
            plt.show()

    def prepare_subgraphs(self):
        self.unique_subgraphs = get_all_unique_graphs(self.settings.subgraph_size)
        letters = choice([letter for letter in ascii_lowercase], len(self.unique_subgraphs), replace=False)
        self.subgraph_letter_map = dict(zip(self.unique_subgraphs, letters))
        self.selected_subgraph_node_ids = self.graph.node_ids[:self.settings.subgraph_size] # TODO - should be connected choice(self.graph.node_ids, self.settings.subgraph_size, replace=False).tolist()

    def prepare_interface(self):
        self.figure = plt.figure("FastText", figsize=(10,10))
        self.figure.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.figure.canvas.mpl_connect('button_press_event', self.on_mouse_press)

        if self.settings.render_isomorphic_graphs:
            div = self.settings.render_x_isomorphisms_per_column
            n = len(self.unique_subgraphs) // div + (len(self.unique_subgraphs) % div > 0)
            grid = Grid(div, div + n, subplot_spec = GridSpec(1, 1)[0])
            self.graph.axis = plt.subplot(grid[:,:div])

            for i, unique_subgraph in enumerate(self.unique_subgraphs):
                unique_subgraph.axis = plt.subplot(grid[i%div, div+i//div])
                unique_subgraph.axis.set_title(self.subgraph_letter_map[unique_subgraph])
                unique_subgraph.render(with_labels=False)
                unique_subgraph.hide_border()
                self.figure.add_subplot(unique_subgraph.axis)
        else:
            grid = Grid(1, 1, subplot_spec = GridSpec(1, 1)[0])
            self.graph.axis = plt.subplot(grid[:,:])

        self.graph.render()
        self.graph.hide_border()
        self.figure.add_subplot(self.graph.axis)

    def prepare_markers(self):
        self.node_markers = {}
        self.highlighted_graph = None

        for node_id in self.graph.node_ids:
            x, y = self.graph.node_id_position_map[node_id]
            text = self.graph.axis.text(x, y, "%s" % (node_id), )
            marker = self.graph.axis.scatter([x], [y], marker='*', c='r', zorder = 2, s = 64)
            self.node_markers[node_id] = [text, marker]

            node_visible =  node_id in self.selected_subgraph_node_ids
            text.set_visible(node_visible)
            marker.set_visible(node_visible)

    def start_auto_walk(self):
        self.block_event = Event()
        self.walk_thread = Thread(target=self.drawing_loop, args=[self.block_event])
        self.walk_thread.start()

    def stop_auto_walk(self):
        self.block_event.set()
        self.walk_thread.join()

    def drawing_loop(self, event: Event):
        while True:
            sleep(1)
            self.do_one_random_walk()
            if event.is_set():
                break

    def on_key_press(self, event):
        stdout.flush()
        if event.key == 'x':
            self.do_one_random_walk()
        elif event.key == 'c':
            self.start_auto_walk()
        elif event.key == 'v':
            self.stop_auto_walk()

    def on_mouse_press(self, event):
        return

    def do_one_random_walk(self):
        possible_targets = self.search_for_walk_targets()
        
        previous_node_id, new_node_id = possible_targets[randint(0, len(possible_targets) - 1)]
        self.selected_subgraph_node_ids.remove(previous_node_id)
        self.selected_subgraph_node_ids.append(new_node_id)

        graph = self.find_isomorphism()
        letter = self.subgraph_letter_map[graph] if graph is not None else ''
        print(letter)

        if self.settings.render:
            self.draw_walk(previous_node_id, new_node_id, graph)            

    def search_for_walk_targets(self):
        possible_targets = []
        selected_neighbors_threshold = 2 if self.settings.subgraph_size >= 3 else 1

        for selected_node_id in self.selected_subgraph_node_ids:
            for connected_node_id in self.graph.node_id_edges_map[selected_node_id]:
                if connected_node_id not in self.selected_subgraph_node_ids:
                    continue
                connected_node_neighbors = self.graph.node_id_edges_map[connected_node_id]
                non_selected_neighbors = list(filter(lambda neighbor:neighbor not in self.selected_subgraph_node_ids, connected_node_neighbors))
                selected_neighbors_count = len(connected_node_neighbors) - len(non_selected_neighbors)

                if selected_neighbors_count >= selected_neighbors_threshold:
                    for non_selected_neighbor in non_selected_neighbors:
                        possible_targets.append((selected_node_id, non_selected_neighbor))
        return possible_targets

    def find_isomorphism(self):
        graph = zeros((self.settings.subgraph_size, self.settings.subgraph_size))
        for i, src_node_id in enumerate(self.selected_subgraph_node_ids):
            for j, dst_node_id in enumerate(self.selected_subgraph_node_ids):
                if dst_node_id in self.graph.node_id_edges_map[src_node_id]:
                    graph[i, j] = 1
                    graph[j, i] = 1

        norm_graph = nauty_normalize(graph)
        iso_graph = list(filter(
            lambda graph_class: 
                array_equal(graph, graph_class.isomorphisms[0]) or \
                array_equal(norm_graph, graph_class.isomorphisms[0]) or \
                array_equal(graph, graph_class.isomorphisms[1]) or \
                array_equal(norm_graph, graph_class.isomorphisms[1]),
            self.subgraph_letter_map.keys()))
        iso_graph = iso_graph[0] if len(iso_graph) else None
        return iso_graph

    def draw_walk(self, previous_node_id, new_node_id, graph):

        text, marker = self.node_markers[previous_node_id]
        text.set_visible(False)
        marker.set_visible(False)

        text, marker = self.node_markers[new_node_id]
        text.set_visible(True)
        marker.set_visible(True)

        if self.highlighted_graph is not None:
            self.highlighted_graph.hide_border()
        if graph is not None:
            graph.highlight()
            self.highlighted_graph = graph

        self.figure.canvas.draw()
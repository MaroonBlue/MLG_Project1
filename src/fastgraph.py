import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec as GridSpec
from matplotlib.gridspec import GridSpecFromSubplotSpec as Grid

from numpy import zeros, array_equal
from threading import Thread, Event
from random import randint, random
from string import ascii_letters
from signal import signal, SIGINT
from numpy.random import choice
from sys import stdout
from time import sleep
from tqdm import tqdm

from src.graph import Graph
from src.isomorphic_graphs import get_all_unique_graphs, nauty_normalize

class FastGraphSettings:
    def __init__(self, 
                 render = False, 
                 render_isomorphic_graphs = False, 
                 render_x_isomorphisms_per_column = 6, 
                 subgraph_size = 3,
                 letters_per_sentence = 25,
                 sentences_to_generate = 1000,
                 spacebar_probability = 0.1,
                 end_of_sentence_symbol = '\n') -> None:

        self.render = render
        self.render_isomorphic_graphs = render_isomorphic_graphs if render else False
        self.render_x_isomorphisms_per_column = render_x_isomorphisms_per_column
        self.subgraph_size = subgraph_size
        self.letters_per_sentence = letters_per_sentence
        self.sentences_to_generate = sentences_to_generate
        self.spacebar_probability = spacebar_probability
        self.end_of_sentence_symbol = end_of_sentence_symbol

        def assert_that(condition, message):
            if not condition:
                raise AssertionError(message)

        assert_that(subgraph_size >= 2 and subgraph_size <= 6, "Subgraph size should be in range [2,6]")
        assert_that(render_x_isomorphisms_per_column >= 1 and render_x_isomorphisms_per_column <= 6, "Isomorphisms/column should be in range [1,6]")
        assert_that(letters_per_sentence >= 1, "Letters per sentence should be in range [1,+oo)")
        assert_that(sentences_to_generate >= 1, "Sentences to generate should be in range [1,+oo)")
        assert_that(spacebar_probability >= 0.0 and spacebar_probability <= 1.0, "Spacebar probability should be in range [0,1]")
        assert_that(end_of_sentence_symbol in ['', ' ', '\n'], "End of sentence symbol should be either '', ' ', '\\n'")

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
            self.draw_initial_walk()
            plt.show()
        else:
            signal(SIGINT, lambda signum, frame: self.stop_auto_walk())
            self.start_auto_walk()

    def prepare_subgraphs(self):
        self.unique_subgraphs = get_all_unique_graphs(self.settings.subgraph_size)
        letters = choice([letter for letter in ascii_letters], len(self.unique_subgraphs), replace=False)
        self.subgraph_letter_map = dict(zip(self.unique_subgraphs, letters))
        self.selected_subgraph_node_ids = self.graph.node_ids[:self.settings.subgraph_size] # TODO - should be connected choice(self.graph.node_ids, self.settings.subgraph_size, replace=False).tolist()

    def prepare_interface(self):
        self.figure = plt.figure("FastText", figsize=(10,10))
        self.figure.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.figure.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.figure.canvas.mpl_connect('close_event', self.stop_auto_walk)


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
        self.node_edges = {}
        self.highlighted_graph = None

        for node_id in self.graph.node_ids:
            x, y = self.graph.node_id_position_map[node_id]

            edges = []
            for target_node_id in self.graph.node_id_edges_map[node_id]:
                xt, yt = self.graph.node_id_position_map[target_node_id]
                dx, dy = xt - x, yt - y 
                arrow = self.graph.axis.arrow(x, y, dx, dy, zorder = 2, color = 'r', visible = False)
                edges.append((target_node_id, arrow))

            text = self.graph.axis.text(x, y, "%s" % (node_id), visible = False)
            marker = self.graph.axis.scatter([x], [y], marker='*', color ='r', zorder = 2, s = 64, visible = False)
            self.node_markers[node_id] = [text, marker, edges]

    def start_auto_walk(self, event = None):
        self.block_event = Event()
        self.walk_thread = Thread(target=self.drawing_loop, args=[self.block_event])
        self.walk_thread.start()

    def stop_auto_walk(self, event = None):
        if self.block_event:
            self.block_event.set()
            self.walk_thread.join()

    def drawing_loop(self, event: Event):
        with open("output.txt", "w") as file:
            for sentence_nr in tqdm(range(self.settings.sentences_to_generate)):

                sentence = ""
                for letter_nr in range(self.settings.letters_per_sentence):
                    sentence += self.do_one_random_walk()
                    if random() < self.settings.spacebar_probability:
                        sentence += " "
                    if self.settings.render:
                        sleep(0.5)
                    if event.is_set():
                        break

                sentence += self.settings.end_of_sentence_symbol
                file.write(sentence)

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

        if self.settings.render:
            self.draw_walk(previous_node_id, new_node_id, graph)

        return letter        

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

    def draw_initial_walk(self):
        for src_node_id in self.selected_subgraph_node_ids:
            text, marker, edges = self.node_markers[src_node_id]
            text.set_visible(True)
            marker.set_visible(True)
            for target_node_id, edge in edges:
                edge.set_visible(target_node_id in self.selected_subgraph_node_ids)
        self.figure.canvas.draw()

    def draw_walk(self, previous_node_id, new_node_id, graph):
        text, marker, edges = self.node_markers[previous_node_id]
        text.set_visible(False)
        marker.set_visible(False)
        _, _, edges = self.node_markers[previous_node_id]
        for _, edge in edges:
            edge.set_visible(False)

        text, marker, edges = self.node_markers[new_node_id]
        text.set_visible(True)
        marker.set_visible(True)
        for node_id in self.selected_subgraph_node_ids:
            _, _, edges = self.node_markers[node_id]
            for target_node_id, edge in edges:
                edge.set_visible(target_node_id in self.selected_subgraph_node_ids)

        if self.highlighted_graph is not None:
            self.highlighted_graph.hide_border()
        if graph is not None and self.settings.render_isomorphic_graphs:
            graph.highlight()
            self.highlighted_graph = graph

        self.figure.canvas.draw()
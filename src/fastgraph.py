import fasttext as f
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec as GridSpec
from matplotlib.gridspec import GridSpecFromSubplotSpec as Grid

from string import ascii_lowercase as ascii_set
from numpy import zeros, array_equal
from threading import Thread, Event
from random import randint, random
from signal import signal, SIGINT
from numpy.random import choice
from numpy.linalg import norm
from sys import stdout
from time import sleep
from tqdm import tqdm
from networkx import degree_centrality, closeness_centrality, betweenness_centrality

from src.graph import Graph
from src.graph_utils import get_all_unique_graphs, nauty_normalize, is_connected

from random import seed as r_seed
from numpy.random import seed as n_seed

r_seed(69)
n_seed(69)

class FastGraphSettings:
    def __init__(self, 
                 render = False, 
                 render_auto_walk_delay_seconds = 0.2,
                 render_isomorphic_graphs = False, 
                 render_x_isomorphisms_per_column = 6, 
                 subgraph_size = 3,
                 letters_per_sentence = 25,
                 sentences_to_generate = 1000,
                 spacebar_probability = 0.1,
                 end_of_sentence_symbol = '\n',
                 file_name = "output") -> None:

        self.render = render
        self.render_auto_walk_delay_seconds = render_auto_walk_delay_seconds
        self.render_isomorphic_graphs = render_isomorphic_graphs if render else False
        self.render_x_isomorphisms_per_column = render_x_isomorphisms_per_column
        self.subgraph_size = subgraph_size
        self.letters_per_sentence = letters_per_sentence
        self.sentences_to_generate = sentences_to_generate
        self.spacebar_probability = spacebar_probability
        self.end_of_sentence_symbol = end_of_sentence_symbol
        self.label = file_name

        def assert_that(condition, message):
            if not condition:
                raise AssertionError(message)

        assert_that(render_auto_walk_delay_seconds >= 0, "Auto walk delay cannot be a negative number")
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
        self.choose_random_start_position()
        self.label = settings.label
        self.model = None

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
        letters = choice([letter for letter in ascii_set], len(self.unique_subgraphs), replace=False)
        self.subgraph_letter_map = dict(zip(self.unique_subgraphs, letters))

    def choose_random_start_position(self):
        starting_node = self.graph.node_ids[randint(0, len(self.graph.node_ids) - 1)]
        neighbors = list(self.graph.node_id_edges_map[starting_node])

        self.selected_subgraph_node_ids = [starting_node]
        for i in range(1000): # Try multiple times and reset if no subgraph can be found
            if len(self.selected_subgraph_node_ids) != self.settings.subgraph_size:
                neighbor_node = neighbors[randint(0, len(neighbors) - 1)]
                if neighbor_node not in self.selected_subgraph_node_ids:
                    self.selected_subgraph_node_ids.append(neighbor_node)
                    neighbors.remove(neighbor_node)
                    neighbors.extend(self.graph.node_id_edges_map[neighbor_node])
                    neighbors = list(set(neighbors))
            else:
                return
        self.choose_random_start_position()

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
        try:
            self.block_event
            if self.block_event.is_set(): raise Exception()
        except:
            self.block_event = Event()
            self.walk_thread = Thread(target=self.walking_loop, args=[self.block_event])
            self.walk_thread.start()

    def stop_auto_walk(self, event = None):
        try:
            self.block_event.set()
            self.walk_thread.join()
        except:
            pass

    def walking_loop(self, event: Event):
        with open(f"outputs/{self.label}.txt", "w") as file:
            for sentence_nr in tqdm(range(self.settings.sentences_to_generate)):
                sentence = ""
                for letter_nr in range(self.settings.letters_per_sentence):
                    sentence += self.do_one_random_walk()
                    if random() < self.settings.spacebar_probability:
                        sentence += " "
                    if self.settings.render:
                        self.graph.axis.set_title(sentence)
                        sleep(self.settings.render_auto_walk_delay_seconds)
                    if event.is_set():
                        break

                sentence += self.settings.end_of_sentence_symbol
                if self.settings.render:
                    self.clear_walk()
                self.choose_random_start_position()
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
        elif event.key == 'z':
            self.toggle_rendering()

    def on_mouse_press(self, event):
        return

    def toggle_rendering(self):
        self.settings.render = not self.settings.render

    def do_one_random_walk(self):
        possible_targets = self.search_for_walk_targets()
        graph = None

        while len(possible_targets):
            possible_target = possible_targets[randint(0, len(possible_targets) - 1) if len(possible_targets) > 1 else 0]
            previous_node_id, new_node_id = possible_target
            graph = self.find_isomorphism(previous_node_id, new_node_id)
            if graph is not None: 
                break
            else:
                possible_targets.remove(possible_target)

        letter = self.subgraph_letter_map[graph] if graph is not None else ''

        if self.settings.render:
            self.draw_walk(previous_node_id, new_node_id, graph)

        return letter 

    def get_node_subgraph_letter(self, node_ids):
        graph = self.find_isomorphism_from_node_ids(node_ids)
        return self.subgraph_letter_map[graph] if graph is not None else ''

    def search_for_walk_targets(self):
        possible_targets = []
        selected_neighbors_threshold = 2 if self.settings.subgraph_size >= 3 else 1

        # For every selected node
        for selected_node_id in self.selected_subgraph_node_ids:
            # Check its connected, selected neighbors
            removal_possible = True
            for connected_node_id in self.graph.node_id_edges_map[selected_node_id]:
                # If the neighbor is selected
                if connected_node_id not in self.selected_subgraph_node_ids:
                    continue
                # Check if the amount of his neighbors is at least 2
                connected_node_neighbors = self.graph.node_id_edges_map[connected_node_id]
                non_selected_neighbors = list(filter(lambda neighbor:neighbor not in self.selected_subgraph_node_ids, connected_node_neighbors))
                selected_neighbors_count = len(connected_node_neighbors) - len(non_selected_neighbors)

                # Then mark the node as removable and add every friendly node for all selected nodes as targets
                if selected_neighbors_count < selected_neighbors_threshold:
                    removal_possible = False
                    break
            # If all neighbors have at least 2 selected neighbors, removal is possible
            if removal_possible:
                # Find all neighbors' neighbors
                nodes_connected_to_neighbors = list(map(lambda node_id: self.graph.node_id_edges_map[node_id] if node_id != selected_node_id else [], self.selected_subgraph_node_ids))
                non_selected_unique_targets = set([i for array in nodes_connected_to_neighbors for i in array])
                
                # Remove all currently selected ones
                for node_id in self.selected_subgraph_node_ids:
                    if node_id in non_selected_unique_targets:
                        non_selected_unique_targets.remove(node_id)
                
                # Add all possibilities to the list
                for unique_target in non_selected_unique_targets:
                    possible_targets.append((selected_node_id, unique_target))

        return possible_targets

    def find_isomorphism(self, previous_node_id, new_node_id):
        adjusted_subgraph_node_ids = self.selected_subgraph_node_ids.copy()
        adjusted_subgraph_node_ids.remove(previous_node_id)
        adjusted_subgraph_node_ids.append(new_node_id)
        
        graph = zeros((self.settings.subgraph_size, self.settings.subgraph_size))
        for i, src_node_id in enumerate(adjusted_subgraph_node_ids):
            for j, dst_node_id in enumerate(adjusted_subgraph_node_ids):
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
        iso_graph = iso_graph if iso_graph is not None and is_connected(iso_graph.isomorphisms[0]) else None
        self.selected_subgraph_node_ids = adjusted_subgraph_node_ids if iso_graph is not None else self.selected_subgraph_node_ids

        return iso_graph
    

    def find_isomorphism_from_node_ids(self, subgraph_node_ids):

        graph = zeros((self.settings.subgraph_size, self.settings.subgraph_size))
        for i, src_node_id in enumerate(subgraph_node_ids):
            for j, dst_node_id in enumerate(subgraph_node_ids):
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
        iso_graph = iso_graph if iso_graph is not None and is_connected(iso_graph.isomorphisms[0]) else None
        self.selected_subgraph_node_ids = subgraph_node_ids if iso_graph is not None else self.selected_subgraph_node_ids

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
        for _, edge in edges:
            edge.set_visible(False)

        text, marker, edges = self.node_markers[new_node_id]
        text.set_visible(True)
        marker.set_visible(True)
        for node_id in self.selected_subgraph_node_ids:
            _, _, edges = self.node_markers[node_id]
            for target_node_id, edge in edges:
                edge.set_visible(target_node_id in self.selected_subgraph_node_ids)

        if self.highlighted_graph is not None and self.settings.render_isomorphic_graphs:
            self.highlighted_graph.hide_border()
        if graph is not None and self.settings.render_isomorphic_graphs:
            graph.highlight()
            self.highlighted_graph = graph

        self.figure.canvas.draw()

    def clear_walk(self):
        for node_id in self.selected_subgraph_node_ids:
            text, marker, edges = self.node_markers[node_id]
            text.set_visible(False)
            marker.set_visible(False)
            for _, edge in edges:
                edge.set_visible(False)

        if self.highlighted_graph is not None and self.settings.render_isomorphic_graphs:
            self.highlighted_graph.hide_border()

        self.figure.canvas.draw()

    def find_centrality_subgraph(self, node_id, tries = 1000):
        
        for i in range(tries):
            starting_node = node_id
            neighbors = list(self.graph.node_id_edges_map[starting_node])
            subgraph_node_ids = [starting_node]
            while len(subgraph_node_ids) != self.settings.subgraph_size:
                neighbor_node = neighbors[randint(0, len(neighbors) - 1)]
                if len(neighbors) + len(subgraph_node_ids) < self.settings.subgraph_size:
                    break
                if neighbor_node not in subgraph_node_ids:
                    subgraph_node_ids.append(neighbor_node)
                    neighbors.remove(neighbor_node)
                    neighbors.extend(self.graph.node_id_edges_map[neighbor_node])
                    neighbors = list(set(neighbors))
            subgraph = self.graph.nx_graph.subgraph(subgraph_node_ids)
            subgraph_centralities = degree_centrality(subgraph)
            if subgraph_centralities[node_id] == max(subgraph_centralities.values()):
                return subgraph

        return subgraph
        raise AssertionError("Failed to generate a centrality graph")
    
    def find_random_subgraph(self, node_id):
        
        starting_node = node_id
        neighbors = list(self.graph.node_id_edges_map[starting_node])
        subgraph_node_ids = [starting_node]
        while len(subgraph_node_ids) != self.settings.subgraph_size:
            neighbor_node = neighbors[randint(0, len(neighbors) - 1)]
            if len(neighbors) + len(subgraph_node_ids) < self.settings.subgraph_size:
                break
            if neighbor_node not in subgraph_node_ids:
                subgraph_node_ids.append(neighbor_node)
                neighbors.remove(neighbor_node)
                neighbors.extend(self.graph.node_id_edges_map[neighbor_node])
                neighbors = list(set(neighbors))
        return self.graph.nx_graph.subgraph(subgraph_node_ids)

    def fit_model(self):
        self.model = f.train_unsupervised(f"outputs/{self.label}.txt", model='skipgram', minCount = 0)
        self.create_node_embeddings()

    def find_node_embedding(self, node_id):
        centrality_subgraph = self.find_centrality_subgraph(node_id)
        letter = self.get_node_subgraph_letter(centrality_subgraph)
        embedding = self.model.get_sentence_vector(letter)
        return embedding

    def find_node_embedding_v2(self, node_id):
        sentence = ""
        for i in range(self.settings.letters_per_sentence):
            centrality_subgraph = self.find_random_subgraph(node_id)
            letter = self.get_node_subgraph_letter(centrality_subgraph)
            sentence += letter
        embedding = self.model.get_sentence_vector(sentence)
        return embedding
    
    def create_node_embeddings(self):
        node_ids = self.graph.node_ids
        embedding_values = [self.find_node_embedding_v2(id) for id in self.graph.node_ids]
        self.embeddings_dict = dict(map(lambda i,j : (i,j) , node_ids, embedding_values))
    
    def get_nearest_neighbors(self, node_id):
        node_embedding = self.embeddings_dict[node_id]
        embedding_items = list(self.embeddings_dict.items())
        embedding_items = list(map(lambda x:[x[0], norm(x[1] - node_embedding)], embedding_items))
        return sorted(embedding_items, key = lambda x:x[1])


        
        



        


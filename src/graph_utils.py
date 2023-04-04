import numpy as np
import oapackage as oa

if __name__ == "__main__":
    from graph import IsomorphismGraph
else:
    from src.graph import IsomorphismGraph

def generate_all_possible_undirected_graphs(n: int):
    # The total number of edges in all generated graphs
    # is equal to the sum of possible counts of '1' in n-sized list,
    # that is sum from 1 to n. This equals n*(n-1)//2.
    # 
    # The number of all possible undirected binary symmetric graphs
    # is therefore 2**(n*(n-1)//2), since the values can be either 0 or 1
    graphs = np.zeros((2 ** ((n-1)*n//2), n, n))
    for i in range(2 ** (n*(n-1)//2)):
        index = 0
        for row in range(n):
            for col in range(row+1,n):
                if (i & (1 << index)) > 0:
                    graphs[i,row,col] = 1
                index += 1
        graphs[i] = graphs[i] + graphs[i].T # make matrix symmetric

    # Returns a matrix of adjacency matrices
    return graphs

def is_connected(adj_matrix: np.ndarray):
    def dfs(graph, visited, node):
        visited[node] = True
        for i in range(len(graph)):
            if graph[node][i] and not visited[i]:
                dfs(graph, visited, i)

    n = adj_matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    dfs(adj_matrix, visited, 0)
    return np.all(visited)

def filter_non_connected_graphs(graphs: np.ndarray):    
    connected_graphs = list(filter(is_connected, graphs))
    return connected_graphs

def nauty_normalize(graph):
    def inverse_permutation(perm):
        inverse = [0] * len(perm)
        for i, p in enumerate(perm):
            inverse[p] = i
        return inverse

    normal_transform = oa.reduceGraphNauty(graph, verbose=0)
    inverse_transform = inverse_permutation(normal_transform)
    graph_normalized = oa.transformGraphMatrix(graph, inverse_transform)

    return graph_normalized

def filter_isomorphic_duplicates(graphs: np.ndarray):  
    unique_normalized_graphs = []
    for graph in graphs:
        graph_normalized = nauty_normalize(graph)
        unique = True
        for unique_norm in unique_normalized_graphs:
            if np.array_equal(unique_norm, graph_normalized):
                unique = False
                break
        if unique:
            unique_normalized_graphs.append(graph_normalized)

    return unique_normalized_graphs

def get_all_unique_graphs(n: int):
    graphs = generate_all_possible_undirected_graphs(n)
    connected_graphs = filter_non_connected_graphs(graphs)
    uniques = filter_isomorphic_duplicates(connected_graphs)

    return list(map(lambda unique : IsomorphismGraph(unique, nauty_normalize(unique)), uniques))

if __name__ == "__main__":
    n = 6
    graphs = generate_all_possible_undirected_graphs(n)
    connected_graphs = filter_non_connected_graphs(graphs)
    unique_graphs = filter_isomorphic_duplicates(connected_graphs)

    search_for = np.array(
        [[0,1,0,0,1],
         [1,0,1,0,0],
         [0,1,0,1,0],
         [0,0,1,0,1],
         [1,0,0,1,0]])
    search_for_reduced = nauty_normalize(search_for)
    # print(search_for_reduced)
    
    def print_stuff(graphs, search_for):
        found = -1
        for i,graph in enumerate(graphs):
            if np.array_equal(graph, search_for):
                found = i
            #print(graph)
        print(f"Total graphs: {len(graphs)}, found: {found}")

    print_stuff(graphs, search_for)
    print_stuff(connected_graphs, search_for)
    print_stuff(unique_graphs, search_for)
    print_stuff(unique_graphs, search_for_reduced)
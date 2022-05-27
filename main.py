import os
import numpy as np
import logging

def create_graphs_from_data(sparse_adjacency, graph_identifiers):
    ''' Creates graphs as array of adjacency matrices
    
    Returns:
        graphs: List of lists containing adjacency matrices
        [[[1, 0, ..., 0],
          [0, 0, ..., 0],
                ...
          [0, 0, ..., 0]],
          ...
          [[1, 0, ..., 0],
          [0, 0, ..., 0],
                ...
          [0, 0, ..., 0]]
        ]
    '''

    # Create list of graph nodes
    graph_nodes = []
    all_graphs_nodes = []
    prev_graph_id = 1
    for node, graph_id in enumerate(graph_identifiers):
        if int(graph_id) == prev_graph_id:
            graph_nodes.append(node+1)
        else:
            all_graphs_nodes.append(graph_nodes)
            graph_nodes = [node+1]
            prev_graph_id = int(graph_id)
    all_graphs_nodes.append(graph_nodes)

    # Create list of adjacency matrices of graphs
    graphs = []
    edges_done = 0
    for graph_nodes in all_graphs_nodes:
        graph = np.zeros((len(graph_nodes), len(graph_nodes)))
        reference_for_indices = min(graph_nodes)
        for edge in sparse_adjacency[edges_done:]:
            nodes = edge.split(',')
            if int(nodes[0]) in graph_nodes:
                graph[int(nodes[0])-reference_for_indices][int(nodes[1])-reference_for_indices] = 1
                edges_done += 1
            else:
                break
        graphs.append(graph)

    return np.array(graphs)

def load_data(dataset_name):
    ''' Reads whole dataset files
    
    Returns:
        sparse_adjacency_file: list of lines - strings
        graph_identifiers_file: list of lines - strings
    '''
    cwd_path = os.getcwd()
    if dataset_name == "imdb-binary":
        dataset_A_path = 'data/IMDB-BINARY/IMDB-BINARY_A.txt'
        dataset_indicator_path = 'data/IMDB-BINARY/IMDB-BINARY_graph_indicator.txt'
    elif dataset_name == "imdb-multi":
        dataset_A_path = 'data/IMDB-MULTI/IMDB-MULTI_A.txt'
        dataset_indicator_path = 'data/MULTI-BINARY/IMDB-MULTI_graph_indicator.txt'
    elif dataset_name == "collab":
        dataset_A_path = 'data/COLLAB/COLLAB_A.txt'
        dataset_indicator_path = 'data/COLLAB/COLLAB_graph_indicator.txt' 
    
    with open(os.path.join(cwd_path, dataset_A_path), 'r') as sparse_adjacency_file, \
        open(os.path.join(cwd_path, dataset_indicator_path), 'r') as  graph_identifiers_file:
        sparse_adjacency = sparse_adjacency_file.read()
        graph_identifiers = graph_identifiers_file.read()

    return sparse_adjacency.strip().split('\n'), graph_identifiers.strip().split('\n')


if __name__ == "__main__":
    datasets = ["imdb-binary", "imdb-multi", "collab"]

    # for dataset in datasets:
    #     sparse_adjacency, graph_identifiers = load_data(dataset)
    #     graphs = create_graphs_from_data(sparse_adjacency, graph_identifiers)

    sparse_adjacency, graph_identifiers = load_data(datasets[0])
    graphs = create_graphs_from_data(sparse_adjacency, graph_identifiers)

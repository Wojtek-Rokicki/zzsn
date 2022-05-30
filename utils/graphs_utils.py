import os
import numpy as np
from utils.dataloaders_utils import split_dataset
import sys

sys.path.append('../')
from global_const import *

def create_graphs_from_data(sparse_adjacency, graph_identifiers):
    ''' Creates graphs as array of adjacency matrices
    
    Returns:
        graphs: List of lists containing adjacency matrices
        [[[1, 0, ..., 0],P
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

    return np.array(graphs,dtype=object)


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
        
    
    with open(os.path.join(ROOT_DIR, dataset_A_path), 'r') as sparse_adjacency_file, \
        open(os.path.join(ROOT_DIR, dataset_indicator_path), 'r') as  graph_identifiers_file:
        sparse_adjacency = sparse_adjacency_file.read()
        graph_identifiers = graph_identifiers_file.read()

    return sparse_adjacency.strip().split('\n'), graph_identifiers.strip().split('\n')


def generate_dataset_graphs(dataset):
    path = ROOT_DIR.joinpath("graphs_data", f'{dataset}')
    path=str(path)
    sparse_adjacency, graph_identifiers = load_data(dataset)
    if not os.path.exists(path + '.npy'):
        graphs = create_graphs_from_data(sparse_adjacency, graph_identifiers)
        np.save(path ,graphs, allow_pickle=True)


def generate_all_datasets_graphs():
    datasets = ["imdb-binary", "imdb-multi", "collab"]
    for dataset in datasets:
        generate_dataset_graphs(dataset)
    

def read_dataset_graphs(dataset): 
    path = ROOT_DIR.joinpath("graphs_data", f'{dataset}')
    path=str(path)
    print(path)
    if not os.path.exists(path + '.npy'):
        generate_dataset_graphs(dataset)
        
    all_graphs = np.load(path + '.npy', allow_pickle=True)
    return all_graphs

def standardize_graphs(all_graphs, max_size=-1):
    ''' Standarizes graphs by extending diagonal with padding margin
        
        Returns:
            List of graphs standarized by diagonal margin padding to the max graph size
    '''
    PADDING_VAL = 0

    # Sorts graphs by size
    lengths = [g.shape[0] for g in all_graphs]
    argsort = np.argsort(lengths)
    all_graphs = [all_graphs[i] for i in argsort]
    
    # Standardizing size of graphs
    if max_size == -1:
        max_size = all_graphs[-1].shape[0]
    else:
        max_size = max_size 

    all_graphs_standardized = []
    for graph in all_graphs:
        # Zeroing left bottom half of symetric adjacency matrix
        graph_size = graph.shape[0]
        for i in range(graph_size):
            for j in range(graph_size):
                if i > j:
                    graph[i, j] = 0

        # Pasting triangle to the right top and left bottom corner
        standardized_graph = np.zeros((max_size, max_size), dtype=np.float32)
        standardized_graph[:(graph_size), (max_size - graph_size):] = graph # top right corner
        graph = graph.transpose()
        standardized_graph[(max_size - graph_size):, :(graph_size)] = graph ## down left corner
        standardized_graph.astype('float32')
        all_graphs_standardized.append(standardized_graph)

    return all_graphs_standardized, max_size

def get_standardized_graphs(dataset, config):
    path = ROOT_DIR.joinpath("standardized_graphs_data", f'{dataset}')
    path = str(path)
    if not os.path.exists(path + '.npy'):
        all_graphs = read_dataset_graphs(dataset)
        train_graphs, test_graphs = split_dataset(all_graphs, config)
        # TODO: Ugly passing max_size of graph between train and test, problem - what if test has max_size matrix
        train_standarized_graphs, max_size = standardize_graphs(train_graphs)
        test_standarized_graphs, _ = standardize_graphs(test_graphs, max_size)
        np.save(path + '_train.npy', train_standarized_graphs, allow_pickle=True)  
        np.save(path + '_test.npy', test_standarized_graphs, allow_pickle=True)
    else:
        train_standarized_graphs = np.load(path + '_train.npy')
        test_standarized_graphs = np.load(path + '_test.npy')
    return train_standarized_graphs, test_standarized_graphs


if __name__ == "__main__":
    generate_all_datasets_graphs()
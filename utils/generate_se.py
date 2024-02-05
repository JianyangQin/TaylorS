import torch
import numpy as np
import os
import argparse
import pickle
import yaml
import csv
import node2vec
import numpy as np
import networkx as nx
from gensim.models import Word2Vec

def inverse_normalize(data):
    zero_mask = data == 0.
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = 1 - (data - min_val) / (max_val - min_val)
    normalized_data[zero_mask] = 0.
    for i in range(len(normalized_data)):
        normalized_data[i, i] = 1.
    return normalized_data

def generate_adj(adj_filename, save_path, num_of_vertices):
    if 'npy' in adj_filename:
        adj_mx = np.load(adj_filename)
        return adj_mx, None
    elif 'npz' in adj_filename:
        adj = np.load(adj_filename)
        A = adj['adjacency']
        distanceA = adj['distance']
        distanceA = inverse_normalize(distanceA)
    elif 'pkl' in adj_filename:
        try:
            with open(adj_filename, 'rb') as f:
                pickle_data = pickle.load(f)
        except UnicodeDecodeError as e:
            with open(adj_filename, 'rb') as f:
                pickle_data = pickle.load(f, encoding='latin1')
        except Exception as e:
            print('Unable to load data ', adj_filename, ':', e)
            raise
        if ('HZ-METRO' in adj_filename) or ('SH-METRO' in adj_filename):
            distanceA = pickle_data
        else:
            _, _, distanceA = pickle_data
    else:
        import csv
        A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
        distanceA = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)

        with open(adj_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[i, j] = 1
                distanceA[i, j] = distance
        distanceA = inverse_normalize(distanceA)

    adj_txt_filename = os.path.join(save_path, 'Adj.txt')

    with open(adj_txt_filename, 'w') as f:
        for i in range(distanceA.shape[0]):
            for j in range(distanceA.shape[1]):
                f.write('{:d} {:d} {:.4f}\n'.format(i, j, distanceA[i, j]))

    return adj_txt_filename


def read_graph(edgelist):
    G = nx.read_edgelist(
        edgelist, nodetype=int, data=(('weight', float),),
        create_using=nx.DiGraph())

    return G


def learn_embeddings(walks, dimensions, output_file):
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(
        walks, vector_size=dimensions, window=10, min_count=0, sg=1,
        workers=8, epochs=iter)
    model.wv.save_word2vec_format(output_file)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='../configs/HZ-METRO/HZMETRO_gman.yaml', type=str,
                        help="configuration file path")
    args = parser.parse_args()

    print('Read configuration file: %s' % (args.config))
    with open(args.config) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    dataset = configs['data_args']['dataset']
    data_filename = configs['data_args']['data_path']
    adj_filename = configs['data_args']['adj_path']
    se_path = configs['base_model_args']['se_path']
    num_of_vertices = configs['data_args']['num_of_vertices']

    abs_path = os.path.abspath(__file__)
    abs_path = abs_path.split('/')
    abs_path.pop(-1)
    abs_path.pop(-1)
    base_path = "/"
    for path in abs_path:
        base_path = os.path.join(base_path, path)
    print("base path: %s\n" % base_path)

    adj_file = os.path.join(base_path, adj_filename)

    data_path = adj_filename.split('/')
    data_path.pop(-1)
    save_path = ""
    for path in data_path:
        save_path = os.path.join(save_path, path)
    save_path = os.path.join(base_path, save_path)
    print("save path: %s\n" % save_path)

    is_directed = True
    p = 2
    q = 1
    num_walks = 100
    walk_length = 80
    dimensions = 64
    window_size = 10
    iter = 1000

    print("Start generate SE ......")

    print("generating adj file ......")
    adj_file = generate_adj(adj_file, save_path, num_of_vertices)

    print("generating graph ......")
    nx_G = read_graph(adj_file)
    G = node2vec.Graph(nx_G, is_directed, p, q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_length)

    print("learning semantic embedding ......")
    save_file = os.path.join(base_path, se_path)
    print("save file: %s\n" % save_file)
    learn_embeddings(walks, dimensions, save_file)

    print("SE finish")
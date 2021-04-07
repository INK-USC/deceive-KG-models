import numpy as np 
import networkx as nx 

def create_vocab_file(npy_graph, output_path):
    vocab = set()
    for triple in npy_graph:
        vocab.add(triple[0])
        # vocab.add(triple[2])

    vocab_list = list(vocab)
    with open(output_path, 'w') as f:
        for v in vocab_list:
            f.write("%s\n" % v)  

def nx2npy(graph, output_path):
    edges = []
    for edge in list(graph.edges.data()):
        edges.append([edge[0], edge[2]['rel'], edge[1]])

    edges = np.array(edges)
    np.save(output_path, edges)
    return edges    

def npy2nx(npy_graph, output_path):
    graph = nx.MultiDiGraph()
    for triple in npy_graph:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # if head == 5306:
        #     print(head)
        graph.add_edge(head, tail, rel = relation)

    nx.write_gpickle(graph, output_path)
    return graph


if __name__=='__main__':
    kg_np = np.load("../data/movie/kg_initial.npy")
    create_vocab_file(kg_np, "new_vocab_movie.txt")
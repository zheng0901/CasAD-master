import pickle

from absl import app, flags

from utils.graphwave.graphwave import *
from utils.sparse_matrix_factorization import *

# flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('cg_emb_dim', 40, 'Cascade graph embedding dimension.')
flags.DEFINE_integer('gg_emb_dim', 40, 'Global graph embedding dimension.')
flags.DEFINE_integer('max_seq', 100, 'Max length of cascade sequence.')
flags.DEFINE_integer('num_s', 2, 'Number of s for spectral graph wavelets.')
flags.DEFINE_integer('observation_time', 1800, 'Observation time.')

# paths
flags.DEFINE_string ('input', './dataset/xovee/', 'Dataset path.')
flags.DEFINE_string ('gg_path', 'global_graph.pkl', 'Global graph path.')


def sequence2list(filename):
    graphs = dict()
    with open(filename, 'r') as f:
        for line in f:
            paths = line.strip().split('\t')[:-1][:FLAGS.max_seq + 1]
            graphs[paths[0]] = list()
            for i in range(1, len(paths)):
                nodes = paths[i].split(':')[0]
                time = paths[i].split(':')[1]
                graphs[paths[0]].append([[int(x) for x in nodes.split(',')], int(time)])

    return graphs


def read_labels(filename):
    labels = dict()
    with open(filename, 'r') as f:
        for line in f:
            id = line.strip().split('\t')[0]
            labels[id] = line.strip().split('\t')[-1]

    return labels


def write_cascade(graphs, labels, id2row, filename, gg_emb, weight=True):
    """
    Input: cascade graphs, global embeddings
    Output: cascade embeddings, with global embeddings appended
    """
    y_data = list()
    keys = list()  # 存储cascade的id
    time_slice_embeddings_all = list()  # 存储每个时间切片的均值池化结果
    initial_node_embeddings_all = list()  # 存储每个级联的初始节点嵌入
    new_node_embeddings_all = list()  # 存储每个时间片新增节点嵌入
    time_slices_all = list()  # 存储每个级联的时间切片列表
    cascade_i = 0
    cascade_size = len(graphs)
    total_time = 0

    # for each cascade graph, generate its embeddings via wavelets
    for key, graph in graphs.items():
        keys.append(key)
        start_time = time.time()
        y = int(labels[key])

        time_slice_nodes = dict()
        time_slice_embeddings = list()
        new_node_embeddings = list()  # 存储每个时间片的新增节点嵌入
        time_slices = list()  # 存储时间切片
        
        # 用于存储历史节点集合
        history_nodes = set()
        # 用于存储上一个时间片的节点集合
        prev_nodes = set()

        # build graph
        g = nx.Graph()
        nodes_index = list()
        list_edge = list()
        times = list()
        t_o = FLAGS.observation_time
        graph = sorted(graph, key=lambda x: x[1])
        
        # 记录初始节点
        initial_node = None
        if graph and graph[0][0]:  # 确保有节点
            initial_node = graph[0][0][0]  # 第一个时间片的第一个节点

        # add edges into graph
        for path in graph:
            t = path[1]
            if t >= t_o:
                continue
            nodes = path[0]
            # 更新历史节点集合
            history_nodes.update(nodes)
            if len(nodes) == 1:
                nodes_index.extend(nodes)
                times.append(1)
                time_slice_nodes[1] = list(history_nodes)
                continue
            else:
                nodes_index.extend([nodes[-1]])
            if weight:
                edge = (nodes[-1], nodes[-2], (1 - t / t_o))  # weighted edge
                times.append(1 - t / t_o)
                time_slice_nodes[1 - t / t_o] = list(history_nodes)
            else:
                edge = (nodes[-1], nodes[-2])
            list_edge.append(edge)

        if weight:
            g.add_weighted_edges_from(list_edge)
        else:
            g.add_edges_from(list_edge)

        # this list is used to make sure the node order of `chi` is same to node order of `cascade`
        nodes_index_unique = list(set(nodes_index))
        nodes_index_unique.sort(key=nodes_index.index)

        # embedding dim check
        d = FLAGS.cg_emb_dim / (2 * FLAGS.num_s)
        if FLAGS.cg_emb_dim % 4 != 0:
            raise ValueError

        # generate cascade embeddings
        chi, _, _ = graphwave_alg(g, np.linspace(0, 100, int(d)),
                                  taus='auto', verbose=False,
                                  nodes_index=nodes_index_unique,
                                  nb_filters=FLAGS.num_s)

        initial_embedding = None
        if initial_node is not None and initial_node in nodes_index_unique:
            initial_embedding = np.hstack([chi[nodes_index_unique.index(initial_node)], 
                                         gg_emb[id2row[initial_node]]])
        else:
            initial_embedding = np.zeros(FLAGS.cg_emb_dim + FLAGS.gg_emb_dim)

        sorted_times = sorted(time_slice_nodes.keys(), reverse=True)
        for t in sorted_times:
            time_slices.append(t)
            
            nodes = list(time_slice_nodes[t])
            if not nodes:
                continue  
            
            current_nodes = set(nodes)
            new_nodes = current_nodes - prev_nodes
            new_nodes = [n for n in new_nodes if n in nodes_index_unique]
            new_node_embedding = [np.hstack([chi[nodes_index_unique.index(node)], gg_emb[id2row[node]]]) 
                                for node in new_nodes]
            
            valid_nodes = [n for n in nodes if n in nodes_index_unique]
            node_embeddings = [np.hstack([chi[nodes_index_unique.index(node)], gg_emb[id2row[node]]]) 
                             for node in valid_nodes]
            mean_embedding = np.mean(node_embeddings, axis=0) if node_embeddings else np.zeros(FLAGS.cg_emb_dim + FLAGS.gg_emb_dim)
            
            time_slice_embeddings.append(mean_embedding)
            
            if new_node_embedding:
                new_node_mean = np.mean(new_node_embedding, axis=0)
                new_node_embeddings.append(new_node_mean)
            else:
                new_node_embeddings.append(np.zeros_like(mean_embedding))
                        
            prev_nodes = current_nodes
        
        time_slice_embeddings_all.append(time_slice_embeddings)
        initial_node_embeddings_all.append(initial_embedding)  # 只存储一次初始节点嵌入
        new_node_embeddings_all.append(new_node_embeddings)
        time_slices_all.append(time_slices)

        # save labels
        y_data.append(y)

        # log
        total_time += time.time() - start_time
        cascade_i += 1
        if cascade_i % 1000 == 0:
            speed = total_time / cascade_i
            eta = (cascade_size - cascade_i) * speed
            print('{}/{}, eta: {:.2f} mins'.format(
                cascade_i, cascade_size, eta/60))

    # write concatenated embeddings into file
    with open(filename, 'wb') as f:
        pickle.dump((keys, time_slices_all, time_slice_embeddings_all, initial_node_embeddings_all, 
                    new_node_embeddings_all, y_data), f)


def main(argv):
    time_start = time.time()
    print('Start to generate graphs and graph embeddings.\n')
    print('Note! This may require a large system memory (~64GB).\n')
    print('Should be finished in about 10-20 minutes.\n')

    # get the information of nodes/users of cascades
    graph_train = sequence2list(FLAGS.input + 'train.txt')
    graph_val = sequence2list(FLAGS.input + 'val.txt')
    graph_test = sequence2list(FLAGS.input + 'test.txt')

    # get the information of labels of cascades
    label_train = read_labels(FLAGS.input + 'train.txt')
    label_val = read_labels(FLAGS.input + 'val.txt')
    label_test = read_labels(FLAGS.input + 'test.txt')

    # load global graph and generate id2row
    with open(FLAGS.input + FLAGS.gg_path, 'rb') as f:
        gg = pickle.load(f)

    # sparse matrix factorization
    print('Generating embeddings of nodes in global graph.')
    model = SparseMatrixFactorization(gg, FLAGS.gg_emb_dim)
    gg_emb = model.pre_factorization(model.matrix, model.matrix)

    ids = [int(xovee) for xovee in gg.nodes()]
    id2row = dict()
    i = 0
    for id in ids:
        id2row[id] = i
        i += 1

    print('Start writing train set into file.')
    write_cascade(graph_train, label_train, id2row, FLAGS.input + 'train.pkl', gg_emb)
    print('Start writing val set into file.')
    write_cascade(graph_val, label_val, id2row, FLAGS.input + 'val.pkl', gg_emb)
    print('Start writing test set into file.')
    write_cascade(graph_test, label_test, id2row, FLAGS.input + 'test.pkl', gg_emb)

    print('Processing time: {:.2f}s'.format(time.time()-time_start))


if __name__ == '__main__':
    app.run(main)

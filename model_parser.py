import argparse

parser = argparse.ArgumentParser()
#  dataset path
parser.add_argument('--input', default='./dataset/weibo/', type=str, help="Dataset path.")
parser.add_argument('--gg_path', default='global_graph.pkl', type=str, help="Global graph path.")

# model
parser.add_argument("--lr", type= float, default=5e-4, help="Learning rate.")
parser.add_argument("--b_size", type=int, default=64, help="Batch size.")
parser.add_argument("--emb_dim", type=int, default=40+40, help="Embedding dimension (cascade emb_dim + global emb_dim")
parser.add_argument("--z_dim", type=int, default=64, help="Dimension of latent variable z.")
parser.add_argument("--rnn_units", type=int, default=128, help="Dimension of latent variable z.")
parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
parser.add_argument("--epochs", type=int, default=1000, help="train epochs.")
parser.add_argument("--max_seq", type=int, default=100, help="Max length of cascade sequence.")

# ODE
parser.add_argument('--ode_units', type=int, default=100, help="Number of units per layer in ODE func")

parser.add_argument("--hidden_dim", type=int, default=64, help="hidden_dim.")

# sliding window
parser.add_argument("--window_size", type=int, default=5, help="Sliding window size for historical state aggregation")

# conv
parser.add_argument('--kernel_h_global', type=int, default=9, help='Global branch convolution kernel height (temporal dimension)')
parser.add_argument('--kernel_w_global', type=int, default=3, help='Global branch convolution kernel width (feature dimension)')
parser.add_argument('--kernel_h_burst', type=int, default=5, help='Burst branch convolution kernel height (temporal dimension)')
parser.add_argument('--kernel_w_burst', type=int, default=3, help='Burst branch convolution kernel width (feature dimension)')
parser.add_argument('--conv_out_channels', type=int, default=64, help='Convolution layer output channels')

parser.add_argument('--spath', default='./checkpoint.pt', type=str, help="Save checkpoint path.")
parser.add_argument("--cu", type=int, default=2, help="select cuda.")
parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
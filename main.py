import os
os.environ["OMP_NUM_THREADS"] = "4"
import argparse
import os.path
from utils import *
from model import *
from Data import *
from tqdm import tqdm


setup_seed(100)
parser = argparse.ArgumentParser(description="use pretraining net work for feature extract")
parser.add_argument("--dataset",
                    dest='dataset',
                    choices=('fashion_mnist',
                             'cifar10',
                             'cifar100',
                             'stl10'),
                    help="Dataset to train",
                    default='stl10')
parser.add_argument("--beta", type=float, default=0)
parser.add_argument("--lambda1", type=float, default=1.0)
parser.add_argument("--lambda2", type=float, default=1.0)
parser.add_argument("--lambda3", type=float, default=1.0)
parser.add_argument("--lambda4", type=float, default=1.0)
parser.add_argument("--lambda5", type=float, default=0.5, help="weight for consistency loss")
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--epochs", type=int, default=150)

args = parser.parse_args()

dim_subspace = 20
ro = 6
alpha = 0.06
num_cluster = 10
num_neighbor = 20
if args.dataset == 'cifar100':
    num_cluster = 100
    args.epochs = 800

# load data
saved_features = torch.load(os.path.join(features_save_dir, args.dataset + features_suffix), weights_only=True)
features = saved_features['data']
label = saved_features['label']

# bulid graph use features
sparse_adj = build_graph(features, K=num_neighbor)
datas = build_pyg_data(features, sparse_adj)

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChebCluster(features=features, hidden_channels=16, num_sample=features[0].shape[0])
model.to(device)

# loss and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


# training loop
pbar = tqdm(range(args.epochs))
for epoch in pbar:
    fusion_expression, content_features, structure_features, content_expression, structure_expression = model(datas)

    # self-expression loss
    attribute_express_loss = F.mse_loss(content_features, torch.mm(content_expression, content_features))
    graph_express_loss = F.mse_loss(structure_features, torch.mm(structure_expression, structure_features))

    # self-expression coefficient loss
    attribute_express_coefficient_loss = torch.linalg.matrix_norm(content_expression, 1)
    graph_express_coefficient_loss = torch.linalg.matrix_norm(structure_expression, 1)

    # C_F loss
    fusion_expression_coefficient_loss = torch.linalg.matrix_norm(fusion_expression, 1)

    # cons_loss
    consistency_loss = F.mse_loss(content_expression, structure_expression)

    # total loss
    total_loss = args.lambda1 * attribute_express_loss + \
                 args.lambda2 * graph_express_loss + \
                 args.lambda3 * attribute_express_coefficient_loss + \
                 args.lambda4 * graph_express_coefficient_loss + \
                 args.lambda5 * consistency_loss +\
                 fusion_expression_coefficient_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    pbar.set_postfix({"loss": total_loss.item()})



print("content self-expression clustering results:")
Ca = content_expression.detach().to('cpu').numpy()
y_pred = sklearn_spectral_clustering(Ca, num_cluster)
print(f"ACC = {cluster_accuracy(label, y_pred):.4f}, NMI = {nmi(label, y_pred):.4f}, ARI = {ari(label, y_pred):.4f}")

print("fusion self-expression clustering results:")
C = fusion_expression.detach().to('cpu').numpy()
y_pred = sklearn_spectral_clustering(C, num_cluster)
print(f"ACC = {cluster_accuracy(label, y_pred):.4f}, NMI = {nmi(label, y_pred):.4f}, ARI = {ari(label, y_pred):.4f}")









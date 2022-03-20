
import torch_geometric.transforms as T
import torch
from utils import ARAPPI

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          add_negative_train_samples=False),
    ])
    dataset = ARAPPI(root='./Data/ara-protein', pre_transform=transform, seq_name="all-MiniLM-L6-v2")
    train_data, val_data, test_data = dataset[0]
    print(train_data)



if __name__ == '__main__':
    main()
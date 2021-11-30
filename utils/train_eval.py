import random
from torch import tensor
from sklearn.model_selection import StratifiedKFold
import torch
import numpy as np
import time
import matplotlib.pyplot as plt


def k_fold(data, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=random.randint(1,999))
    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(data.x.shape[0]), data.y):
        test_indices.append(torch.from_numpy(idx).to(torch.long))
    val_indices = [test_indices[i - 1] for i in range(folds)]
    for i in range(folds):
        train_mask = torch.ones(data.x.shape[0], dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))
    return train_indices, test_indices, val_indices


def cross_validation_with_val_set(data,model,args,transform):
    train_losses, val_aucs, test_aucs, durations = [], [], [], []
    folds = args.folds

    for fold, (train_idx, test_idx,
               val_idx) in enumerate(zip(*k_fold(data, folds))):

        print(f"{fold+1} fold train")
        split = {
            'train_idx': np.array(train_idx),
            'val_idx': np.array(val_idx),
            'test_idx': np.array(test_idx)}
        allmask = {}
        for name in ['train', 'val', 'test']:
            idx = split[f'{name}_idx']
            idx = torch.from_numpy(idx).to(torch.long)
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[idx] = True
            allmask[f'{name}_mask'] = mask

        data.train_mask = allmask['train_mask']
        data.val_mask = allmask['val_mask']
        data.test_mask = allmask['test_mask']

        train_data, val_data, test_data = transform(data)  # Explicitly transform data.
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        def train():
            model.train()
            optimizer.zero_grad()
            z = model.encode(train_data.x, train_data.edge_index)
            loss = model.recon_loss(z, train_data.pos_edge_label_index)
            loss.backward()
            optimizer.step()
            return float(loss)

        @torch.no_grad()
        def test(data):
            model.eval()
            z = model.encode(data.x, data.edge_index)
            return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)

        t_start = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            los = train()
            train_losses.append(los)
            auc, ap = test(test_data)
            test_aucs.append(auc)
            if epoch % args.epochs == 0:
                print('Epoch: {:03d}, Test AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
        t_end = time.perf_counter()
        durations.append(t_end - t_start)

        fig = plt.figure(figsize=(10, 8))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Talidation loss')
        # find position of lowest validation loss
        minposs = train_losses.index(min(train_losses)) + 1
        plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')
        plt.xlabel('epochs')
        plt.ylabel('values')
        # plt.ylim(0.7, 1)  # consistent scale
        # plt.xlim(0, len(val_aucs) + 1)  # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

    loss, auc, duration = tensor(train_losses), tensor(test_aucs), tensor(durations)
    loss, auc = loss.view(folds, args.epochs), auc.view(folds, args.epochs)
    loss, argmin = loss.min(dim=1)
    auc = auc[torch.arange(folds, dtype=torch.long), argmin]
    loss_mean = loss.mean().item()
    auc_mean = auc.mean().item()
    auc_std = auc.std().item()
    duration_mean = duration.mean().item()
    print('Train Loss: {:.4f}, Test AUC: {:.4f} ± {:.3f}, Duration: {:.4f}'.
          format(loss_mean, auc_mean, auc_std, duration_mean))
    return loss_mean, auc_mean, auc_std
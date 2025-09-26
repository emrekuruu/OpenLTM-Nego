from data_provider.data_loader import UnivariateDatasetBenchmark, MultivariateDatasetBenchmark, Global_Temp, Global_Wind, Dataset_ERA5_Pretrain, Dataset_ERA5_Pretrain_Test, UTSD, UTSD_Npy, NegoCompletionDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
from torch.nn.utils.rnn import pad_sequence

def nego_completion_collate_fn(batch):
    batch_x, batch_y, batch_x_mark, batch_y_mark = zip(*batch)

    # Pad sequences to max length in batch (capped at 10000)
    max_x_len = min(max(x.shape[0] for x in batch_x), 10000)
    max_y_len = min(max(y.shape[0] for y in batch_y), 10000)

    # Pad x sequences
    padded_x = []
    for x in batch_x:
        if x.shape[0] > 10000:
            x = x[:10000]
        pad_len = max_x_len - x.shape[0]
        if pad_len > 0:
            padding = torch.zeros(pad_len, x.shape[1], dtype=x.dtype)
            x = torch.cat([x, padding], dim=0)
        padded_x.append(x)

    # Pad y sequences
    padded_y = []
    for y in batch_y:
        if y.shape[0] > 10000:
            y = y[:10000]
        pad_len = max_y_len - y.shape[0]
        if pad_len > 0:
            padding = torch.zeros(pad_len, y.shape[1], dtype=y.dtype)
            y = torch.cat([y, padding], dim=0)
        padded_y.append(y)

    # Pad marks similarly
    padded_x_mark = []
    for x_mark in batch_x_mark:
        if x_mark.shape[0] > 10000:
            x_mark = x_mark[:10000]
        pad_len = max_x_len - x_mark.shape[0]
        if pad_len > 0:
            padding = torch.zeros(pad_len, x_mark.shape[1], dtype=x_mark.dtype)
            x_mark = torch.cat([x_mark, padding], dim=0)
        padded_x_mark.append(x_mark)

    padded_y_mark = []
    for y_mark in batch_y_mark:
        if y_mark.shape[0] > 10000:
            y_mark = y_mark[:10000]
        pad_len = max_y_len - y_mark.shape[0]
        if pad_len > 0:
            padding = torch.zeros(pad_len, y_mark.shape[1], dtype=y_mark.dtype)
            y_mark = torch.cat([y_mark, padding], dim=0)
        padded_y_mark.append(y_mark)

    return (torch.stack(padded_x), torch.stack(padded_y),
            torch.stack(padded_x_mark), torch.stack(padded_y_mark))

data_dict = {
    'UnivariateDatasetBenchmark': UnivariateDatasetBenchmark,
    'MultivariateDatasetBenchmark': MultivariateDatasetBenchmark,
    'Global_Temp': Global_Temp,
    'Global_Wind': Global_Wind,
    'Era5_Pretrain': Dataset_ERA5_Pretrain,
    'Era5_Pretrain_Test': Dataset_ERA5_Pretrain_Test,
    'Utsd': UTSD,
    'Utsd_Npy': UTSD_Npy,
    'NegoCompletionOracle': NegoCompletionDataset,
    'NegoCompletionOpponentModel': NegoCompletionDataset
}


def data_provider(args, flag):
    Data = data_dict[args.data]

    if flag in ['test', 'val']:
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size

    if flag in ['train', 'val']:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.input_token_len, args.output_token_len],
            nonautoregressive=args.nonautoregressive,
            test_flag=args.test_flag,
            subset_rand_ratio=args.subset_rand_ratio
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.test_seq_len, args.input_token_len, args.test_pred_len],
            nonautoregressive=args.nonautoregressive,
            test_flag=args.test_flag,
            subset_rand_ratio=args.subset_rand_ratio
        )
    print(flag, len(data_set))

    # Use custom collate function for negotiation completion datasets
    collate_fn = nego_completion_collate_fn if args.data in ['NegoCompletionOracle', 'NegoCompletionOpponentModel'] else None

    if args.ddp:
        train_datasampler = DistributedSampler(data_set, shuffle=shuffle_flag)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            sampler=train_datasampler,
            num_workers=args.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=drop_last,
            collate_fn=collate_fn
        )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=drop_last,
            collate_fn=collate_fn
        )
    return data_set, data_loader

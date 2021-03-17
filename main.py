import numpy as np
import torch
import argparse
from utils import str2bool
from solver import Solver
import joblib
import json
import pathlib
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    """
    construct dataset from numpy arrays
    
    """
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)

def main(args):
    """"
    main function
    """
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)
    
    print()
    print('[ARGUMENTS]')
    print(args)
    print()

    net = Solver(args)
    
    # create your dataloader
    if args.mode == 'train': net.train()
 
    elif args.mode == 'validate' : net.validate(save_ckpt=True)
    elif args.mode == 'test' : net.test(save_ckpt=False)
    else : return 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='VIB for neural networks')
    parser.add_argument('--data', default = 'urbansound8k', type=str, help='input data source used')
    parser.add_argument('--dim_input', default = 0, type=int, help='input dimension')
    parser.add_argument('--output_features', default = 0, type=int, help='ioutput features number ')
    parser.add_argument('--epoch', default = 100, type=int, help='epoch size')
    parser.add_argument('--lr', default = 1e-4, type=float, help='learning rate')
    parser.add_argument('--beta', default = 1e-3, type=float, help='beta')
    parser.add_argument('--K', default = 256, type=int, help='dimension of encoding Z')
    parser.add_argument('--seed', default = 1, type=int, help='random seed')
    parser.add_argument('--num_avg', default = 1, type=int, help='the number of samples when\
            performing multi-shot prediction')
    parser.add_argument('--dataset', default= '', type=str, help='dataset name')
    parser.add_argument('--train_dataset_percentage', default=1.0, type=float, help='train_dataset_percentage')
    parser.add_argument('--batch_size', default = 32, type=int, help='batch size')
    parser.add_argument('--env_name', default='main', type=str, help='visdom env name')
    parser.add_argument('--dset_dir', default='joblib_features', type=str, help='dataset directory path')
    parser.add_argument('--summary_dir', default='summary', type=str, help='summary directory path')
    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory path')
    parser.add_argument('--load_ckpt',default='', type=str, help='checkpoint name')
    parser.add_argument('--cuda',default=False, type=str2bool, help='enable cuda')
    parser.add_argument('--mode',default='train', type=str, help='train or test')
    parser.add_argument('--tensorboard',default=False, type=str2bool, help='enable tensorboard')
    
    args = parser.parse_args()
    ### create data loader
    if args.data=='urbansound8k': 
        X,y = joblib.load('./joblib_features/Xurbansound8k.joblib'),joblib.load('./joblib_features/yurbansound8k.joblib')
    elif args.data=='emotiontoronto':
        X,y = joblib.load('./joblib_features/X.joblib'),joblib.load('./joblib_features/y.joblib')
    elif args.data=='audioMNIST':
        X,y = joblib.load('./joblib_features/XaudioMNIST.joblib'),joblib.load('./joblib_features/yaudioMNIST.joblib') 
    ## dimension input
    args.dim_input = X.shape[1]
    args.output_features = len(set(y))
    full_dataset = CustomDataset(X, y)
    train_size = int(0.6 * len(full_dataset))
    valid_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - valid_size-train_size
    #train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, 
                                                            #[train_size, test_size])
    ## split the whole dataset into three sets: train, validation, and test                            
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, (train_size, valid_size, test_size), generator=torch.Generator().manual_seed(args.seed))
    
    train_dataset_inuse_size=int(args.train_dataset_percentage * train_size)
    train_dataset_unused_size=train_size- train_dataset_inuse_size
    train_dataset_inuse,train_dataset_unused=torch.utils.data.random_split(
    train_dataset, (train_dataset_inuse_size,train_dataset_unused_size), generator=torch.Generator().manual_seed(args.seed))
    print(train_dataset_inuse_size)
    train_dataloader = DataLoader(train_dataset_inuse, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    my_dataloader = {'train': train_dataloader , 'validate':valid_dataloader, 'test': test_dataloader}
    
    ###
    args.dataset = my_dataloader
    main(args)
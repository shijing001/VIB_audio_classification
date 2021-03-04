import numpy as np
import torch
import argparse
from utils import str2bool
from solver import Solver
import joblib
from torch.utils.data import Dataset, DataLoader

bs=16

class CustomDataset(Dataset):
    """
    construct dataset from numpy
    
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
    
    
X = joblib.load('./joblib_features/X.joblib')
y = joblib.load('./joblib_features/y.joblib')
full_dataset = CustomDataset(X, y)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, 
                                                            [train_size, test_size])
                                                              

train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=True, num_workers=0)
my_dataloader = {'train': train_dataloader , 'test': test_dataloader}





def main(args):


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
    if args.mode == 'train' : net.train()
    elif args.mode == 'test' : net.test(save_ckpt=False)
    else : return 0

    
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='TOY VIB')
    parser.add_argument('--epoch', default = 1, type=int, help='epoch size')
    parser.add_argument('--lr', default = 1e-4, type=float, help='learning rate')
    parser.add_argument('--beta', default = 1e-3, type=float, help='beta')
    parser.add_argument('--K', default = 256, type=int, help='dimension of encoding Z')
   
    parser.add_argument('--seed', default = 1, type=int, help='random seed')
    parser.add_argument('--num_avg', default = 0, type=int, help='the number of samples when\
            perform multi-shot prediction')
    parser.add_argument('--batch_size', default = bs, type=int, help='batch size')
    parser.add_argument('--env_name', default='main', type=str, help='visdom env name')
    parser.add_argument('--dataset', default= my_dataloader, type=str, help='dataset name')
    parser.add_argument('--dset_dir', default='joblib_features', type=str, help='dataset directory path')
    parser.add_argument('--summary_dir', default='summary', type=str, help='summary directory path')
    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory path')
    parser.add_argument('--load_ckpt',default='', type=str, help='checkpoint name')
    parser.add_argument('--cuda',default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--mode',default='train', type=str, help='train or test')
    parser.add_argument('--tensorboard',default=False, type=str2bool, help='enable tensorboard')
    args = parser.parse_args()
    main(args)

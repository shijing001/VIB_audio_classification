import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
#from utils import cuda
import pdb
import time
from numbers import Number
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import sklearn.metrics
import argparse
import json
import pathlib
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class ToyNet(nn.Module):
    '''
    Construct a MLP that is used to train the model 
    param[in]: X_train, output_features
    param[out]: output 
    
    Note: initialize the weight with a self-defined method
    '''

    def __init__(self, args):
        super(ToyNet, self).__init__()
        self.dim_input=args.dim_input
        self.output_features=args.output_features
        self.encode = nn.Sequential(
            nn.Linear(self.dim_input, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.output_features))

    def forward(self, X_train):
        output=self.encode(X_train)
        #prediction = F.softmax(output,dim=1).max(1)[1]
        #print(prediction)
        return output
    
    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])
            
def xavier_init(ms):
    """
    Xavier initialization
    """
    for m in ms :
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight,gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()
            

def cuda(tensor, is_cuda):
    if is_cuda : return tensor.cuda()
    else : return tensor
    
class CustomDataset(Dataset):
    """
    construct dataset from numpy and split it 
    
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

    
class Solver(object):
    
    #train the model
    def __init__(self, args):
        """
        initialization of a Solver object
        
        :params[in]: args, an argparse object
        
        """
        ##__init__
        self.args = args
        self.cuda = (args.cuda and torch.cuda.is_available())
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.num_avg = args.num_avg
        self.train_dataset_percentage=args.train_dataset_percentage
        self.global_iter = 0
        self.global_epoch = 0
        ## Network & Optimizer
        self.toynet = cuda(ToyNet(self.args), self.cuda)
        self.toynet.weight_init()
        self.optim = optim.Adam(self.toynet.parameters(),
                                lr=self.lr,
                                betas=(0.5,0.999))
        self.criterion = nn.CrossEntropyLoss()
        ### load checkpoints
        self.ckpt_dir = Path(args.ckpt_dir).joinpath(args.env_name)
        if not self.ckpt_dir.exists() : self.ckpt_dir.mkdir(parents=True,exist_ok=True)
        self.load_ckpt = args.load_ckpt
        
        if self.load_ckpt != '' : self.load_checkpoint(self.load_ckpt)
        
        ### dataset
        self.data_loader =  args.dataset
        # History
        self.history = dict()
        self.history['avg_acc']=0.
        self.history['f1_score']=0.
        self.history['info_loss']=0.
        self.history['class_loss']=0.
        self.history['total_loss']=0.
        self.history['epoch']=0
        self.history['iter']=0
        
    def set_mode(self, mode='train'):
        if mode == 'train' :
            self.toynet.train()
            #self.toynet_ema.model.train()
        elif mode == 'eval' :
            self.toynet.eval()
            #self.toynet_ema.model.eval()
        else : raise('mode error. It should be either train, or eval')
    
    def train(self):
        self.set_mode('train')
        baseline_train = {"Accuracy":[],"F1_Score":[]}
        baseline_valid = {"Accuracy":[],"F1_Score":[]}
        for epc in range(self.epoch):  # loop over the dataset multiple times
            self.toynet.train('True')  # training neural networks mode
            running_loss = 0.0
            correct = 0
            total_num=0
            accum_accuracy=0
            counter=0
            y_real=torch.randn([0])
            y_hat=torch.randn([0])
            for i, data in enumerate(self.data_loader['train']):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                self.optim.zero_grad()

                # forward + backward + optimize
                outputs = self.toynet.forward(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optim.step()
                
                total_num += labels.size(0)
                prediction = F.softmax(outputs,dim=1).max(1)[1]
                y_real=torch.cat([y_real,labels],dim=0)
                y_hat=torch.cat([y_hat,prediction],dim=0)
                ##total num of correct predictions
                correct += torch.eq(prediction,labels).float().sum()

                # print statistics
                running_loss += loss.item()
                #if i % self.batch_size == 0:   
                
                
            accuracy = sklearn.metrics.accuracy_score(y_real,y_hat)
            f1score = sklearn.metrics.f1_score(y_real, y_hat,labels=None,pos_label=1, average='macro',sample_weight=None)
            accum_accuracy+=accuracy
            avg_accuracy= accum_accuracy/self.global_epoch
            print('[%d, %5d] loss: %.3f' %
                    (epc + 1, i + 1, running_loss / self.batch_size))
            print('acc:{:.4f} '
                    .format(accuracy.item(), end=' '))
            print('err:{:.4f} '
                    .format(1-accuracy.item()))
                    
                    
                    
                    
            baseline_train["Accuracy"].append(float("{:.2f}".format(accuracy.item())))
            baseline_train["F1_Score"].append(float("{:.2f}".format(f1score.item())))
            ## validation set at each epoch
            temp_accuracy,temp_f1score = self.validate()
            ##input accuracy and f1-score of validation dataset into ano
            baseline_valid["F1_Score"].append(float("{:.2f}".format(temp_f1score)))
            baseline_valid["Accuracy"].append(float("{:.2f}".format(temp_accuracy)))
            
        working_dir_path = pathlib.Path().absolute()
        SAVE_DIR_PATH = str(working_dir_path) + '/Dictionaries/baseline'
        fileName1 ='baseline_train'+str(self.train_dataset_percentage)
        fileName2 ='baseline_valid'+str(self.train_dataset_percentage)
        writeToJSONFile(SAVE_DIR_PATH,fileName1,baseline_train)
        writeToJSONFile(SAVE_DIR_PATH,fileName2,baseline_valid)
        print(len(baseline_train),baseline_train)
        print(len(baseline_valid),baseline_valid)
        print('Finished Training',(epc+1))
        
        return baseline_train, baseline_valid
    
    def validate(self,save_ckpt=True):
        self.set_mode('eval')
        """
        Testing over a dataset
        """
        self.toynet.train('False')  # evaluation mode      
        loss, correct, total_num = 0,0,0
        correct = 0
        avg_correct = 0
        total_num = 0
        counter=0
        accum_accuracy =0
        y_real=torch.randn([0])
        y_hat=torch.randn([0])
        
        for i, data in enumerate(self.data_loader['validate']):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            outputs = self.toynet.forward(inputs)
            # loss
            total_num += labels.shape[0]
            loss += self.criterion(outputs,labels)
            prediction = F.softmax(outputs,dim=1).max(1)[1]
            y_real=torch.cat([y_real,labels],dim=0)
            y_hat=torch.cat([y_hat,prediction],dim=0)
            correct += torch.eq(prediction,labels).float().sum()
            
           
                
        accuracy = sklearn.metrics.accuracy_score(y_real,y_hat)
        f1score = sklearn.metrics.f1_score(y_real, y_hat,labels=None,pos_label=1, average='macro',sample_weight=None)
        accum_accuracy+=accuracy
        avg_accuracy= accum_accuracy/self.global_epoch
        
        print('[Validation RESULT]')
        print('acc:{:.4f} '
                .format(accuracy.item(),end=' '))
        print('err:{:.4f}'
                .format(1-accuracy.item()))
        print(classification_report(y_real,y_hat))
        
        if self.history['f1_score'] <f1score.item():
            print('update new params')
            self.history['avg_acc'] = avg_accuracy.item()
            self.history['f1_score'] = f1score.item()
            self.history['loss'] = loss.item()
            self.history['epoch'] = self.global_epoch
            self.history['iter'] = self.global_iter
            if (save_ckpt) :
                {self.save_checkpoint('best_acc.tar'),
                 print("save checkpoint")}
        '''
        if self.history['avg_acc'] < avg_accuracy.item() :
            self.history['avg_acc'] = avg_accuracy.item()
            self.history['class_loss'] = class_loss.item()
            self.history['info_loss'] = info_loss.item()
            self.history['total_loss'] = total_loss.item()
            self.history['epoch'] = self.global_epoch
            self.history['iter'] = self.global_iter
            if (save_ckpt) :
                {self.save_checkpoint('best_acc.tar'),
                 print("save checkpoint")}
        '''
        self.toynet.train('True')
        
        return accuracy.item(), f1score.item()
    def test(self,save_ckpt=True):
        """
        Testing over a dataset
        """
        self.toynet.train('False')  # evaluation mode      
        loss, correct, total_num = 0,0,0
        
        y_real=torch.randn([0])
        y_hat=torch.randn([0])
        ## load the saved params 
        self.load_checkpoint(filename='best_acc.tar')
        ###
        for i, data in enumerate(self.data_loader['test']):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            outputs = self.toynet.forward(inputs)
            # loss
            total_num += labels.shape[0]
            loss += self.criterion(outputs,labels)
            prediction = F.softmax(outputs,dim=1).max(1)[1]
            y_real=torch.cat([y_real,labels],dim=0)
            y_hat=torch.cat([y_hat,prediction],dim=0)
            correct += torch.eq(prediction,labels).float().sum()
            
                
        accuracy = sklearn.metrics.accuracy_score(y_real,y_hat)
        f1score = sklearn.metrics.f1_score(y_real, y_hat,labels=None, average='macro',sample_weight=None)
        
            
        print('[TEST RESULT]')
        print('acc:{:.4f} '
                .format(accuracy.item(),end=' '))
        print('err:{:.4f}'
                .format(1-accuracy.item()))
        print(classification_report(y_real,y_hat))
        
        
      
    
    def save_checkpoint(self, filename='best_acc.tar'):
        model_states = {
                'net':self.toynet.state_dict()
                }
        optim_states = {
                'optim':self.optim.state_dict(),
                }
        states = {
                'iter':self.global_iter,
                'epoch':self.global_epoch,
                'history':self.history,
                'args':self.args,
                'model_states':model_states,
                'optim_states':optim_states,
                }

        file_path = self.ckpt_dir.joinpath(filename)
        torch.save(states,file_path.open('wb+'))
        print("=> saved checkpoint '{}' (iter {})".format(file_path,self.global_iter)) 

    def load_checkpoint(self, filename='best_acc.tar'):
        file_path = self.ckpt_dir.joinpath(filename)
        if file_path.is_file():
            print("=> loading checkpoint '{}'".format(file_path))
            checkpoint = torch.load(file_path.open('rb'))
            self.global_epoch = checkpoint['epoch']
            self.global_iter = checkpoint['iter']
            self.history = checkpoint['history']

            self.toynet.load_state_dict(checkpoint['model_states']['net'])
            
            print("=> loaded checkpoint '{} (iter {})'".format(
                file_path, self.global_iter))

        else:
            print("=> no checkpoint found at '{}'".format(file_path))
            
def writeToJSONFile(path, fileName, data):
    filePathNameWExt =  path + '/' + fileName + '.json'
    with open(filePathNameWExt, 'w') as fp:
        json.dump(data, fp)
    
    
    def str2bool(v):
        """
        codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
        """

        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
            
            
### main function -----
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='MLP baseline model')
    parser.add_argument('--data', default = 'urbansound8k', type=str, help='input data source used')
    parser.add_argument('--dim_input', default = 0, type=int, help='input dimension')
    parser.add_argument('--output_features', default = 0, type=int, help='ioutput features number ')
    parser.add_argument('--epoch', default = 100, type=int, help='epoch size')
    parser.add_argument('--lr', default = 1e-4, type=float, help='learning rate')
    parser.add_argument('--seed', default = 1, type=int, help='random seed')
    parser.add_argument('--num_avg', default = 1, type=int, help='the number of samples when\
            performing multi-shot prediction')
    parser.add_argument('--dataset', default= '', type=str, help='dataset name')
    parser.add_argument('--train_dataset_percentage', default=0.6, type=float, help='train_dataset_percentage')
    parser.add_argument('--batch_size', default = 32, type=int, help='batch size')
    parser.add_argument('--env_name', default='main', type=str, help='visdom env name')
    #parser.add_argument('--dset_dir', default='joblib_features', type=str, help='dataset directory path')
    parser.add_argument('--summary_dir', default='summary', type=str, help='summary directory path')
    
    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory path')
    parser.add_argument('--load_ckpt',default='', type=str, help='checkpoint name')
    parser.add_argument('--cuda',default=False, type=bool, help='enable cuda')
    parser.add_argument('--mode',default='train', type=str, help='train or test')
    parser.add_argument('--tensorboard',default=False, type=bool , help='enable tensorboard')
    
    args = parser.parse_args()
    ### create data loader
    if args.data=='urbansound8k': 
        X,y = joblib.load('./joblib_features/Xurbansound8k.joblib'), joblib.load('./joblib_features/yurbansound8k.joblib')
    elif args.data=='emotiontoronto':
        X,y = joblib.load('./joblib_features/X.joblib'),joblib.load('./joblib_features/y.joblib')
    elif args.data=='audioMNIST':
        X,y = joblib.load('./joblib_features/XaudioMNIST.joblib'),joblib.load('./joblib_features/yaudioMNIST.joblib') 
    ## dimension input
    args.dim_input = X.shape[1]
    ## number of classes
    args.output_features = len(set(y))
    ## construct dataset in a pytorch way
    full_dataset = CustomDataset(X, y)
    train_size,valid_size = int(0.6 * len(full_dataset)),int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - valid_size-train_size
    ## random split data into three sets
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, (train_size, valid_size, test_size), generator=torch.Generator().manual_seed(args.seed))
    ## split the training dataset into two parts
    train_dataset_inuse_size=int(args.train_dataset_percentage * train_size)
    train_dataset_unused_size=train_size- train_dataset_inuse_size
    train_dataset_inuse,train_dataset_unused=torch.utils.data.random_split(
        train_dataset, (train_dataset_inuse_size,train_dataset_unused_size), generator=torch.Generator().manual_seed(args.seed))
    ## print(train_dataset_inuse_size) ## -- to debug
    ## create Dataloaders
    train_dataloader = DataLoader(train_dataset_inuse, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    my_dataloader = {'train': train_dataloader , 'validate':valid_dataloader, 'test': test_dataloader}
    ###
    args.dataset = my_dataloader
    ## instantiate an object
    net=Solver(args)

    ##  # create your dataloaderß
    if (args.mode == 'train'):
        net=Solver(args)
        baseline_train, baseline_valid= net.train()
 
    elif args.mode == 'validate' : net.validate(save_ckpt=True)
    elif args.mode == 'test' : net.test(save_ckpt=False)
    # Example
    #data = baseline_train

    #writeToJSONFile(SAVE_DIR_PATH,fileName,data)

    #fileName ='baseline_valid'+str(args.train_dataset_percentage)
    
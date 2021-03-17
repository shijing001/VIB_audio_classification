import numpy as np
import torch
import argparse
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sklearn.metrics
import json
import pathlib
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from utils import cuda, Weight_EMA_Update
#from datasets.datasets import return_data
from model import ToyNet
from pathlib import Path
from sklearn.metrics import classification_report




class Solver(object):

    def __init__(self, args):
        self.args = args
        self.data = args.data
        self.cuda = (args.cuda and torch.cuda.is_available())
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.eps = 1e-9
        self.K = args.K
        self.train_dataset_percentage=args.train_dataset_percentage
        self.dim_input =args.dim_input
        self.output_features=args.output_features
        self.beta = args.beta
        self.num_avg = args.num_avg
        self.global_iter = 0
        self.global_epoch = 0
        
        # Network & Optimizers
        self.toynet = cuda(ToyNet(self.K,self.args), self.cuda)
        self.toynet.weight_init()
        self.toynet_ema = Weight_EMA_Update(cuda(ToyNet(self.K,self.args),self.cuda), self.toynet.state_dict(), decay=0.999)

        self.optim = optim.Adam(self.toynet.parameters(),lr=self.lr,betas=(0.5,0.999))
        #self.scheduler = lr_scheduler.ExponentialLR(self.optim,gamma=0.97)

        self.ckpt_dir = Path(args.ckpt_dir).joinpath(args.env_name)
        if not self.ckpt_dir.exists() : self.ckpt_dir.mkdir(parents=True,exist_ok=True)
        self.load_ckpt = args.load_ckpt
        if self.load_ckpt != '' : self.load_checkpoint(self.load_ckpt)

        # History
        self.history = dict()
        self.history['avg_acc']=0.
        self.history['f1_score']=0.
        self.history['info_loss']=0.
        self.history['class_loss']=0.
        self.history['total_loss']=0.
        self.history['epoch']=0
        self.history['iter']=0

        # Tensorboard
        self.tensorboard = args.tensorboard
        if self.tensorboard :
            self.env_name = args.env_name
            self.summary_dir = Path(args.summary_dir).joinpath(args.env_name)
            if not self.summary_dir.exists() : self.summary_dir.mkdir(parents=True,exist_ok=True)
            self.tf = SummaryWriter(log_dir=self.summary_dir)
            self.tf.add_text(tag='argument',text_string=str(args),global_step=self.global_epoch)

        # Dataset
        self.data_loader =  args.dataset
        print(self.data_loader)

    def set_mode(self, mode='train'):
        if mode == 'train' :
            self.toynet.train()
            self.toynet_ema.model.train()
        elif mode == 'eval' :
            self.toynet.eval()
            self.toynet_ema.model.eval()
        else : raise('mode error. It should be either train, or eval')

    def train(self):
        self.set_mode('train')
        vib_train = {"Accuracy":[],"F1_Score":[]}
        vib_valid = {"Accuracy":[],"F1_Score":[]}
      
        for e in range(self.epoch):  ## each epoch
            self.global_epoch += 1
            y_real=torch.randn([0])
            y_hat=torch.randn([0])
            correct=0
            total_num=0
            accum_accuracy=0
            counter=0
            print('epoch. ', self.global_epoch)
            for idx, (data,label) in enumerate(self.data_loader['train']):
                counter+=1
                x = Variable(cuda(data, self.cuda))
                y = Variable(cuda(label, self.cuda))
                (mu, std), logit = self.toynet(x)
                total_num+=label.shape[0]
                
                class_loss = F.cross_entropy(logit,y).div(math.log(2))
                info_loss = -0.5*(1+2*std.log()-mu.pow(2)-std.pow(2)).sum(1).mean().div(math.log(2))
                total_loss = class_loss + self.beta*info_loss
                #print("class_loss is,",class_loss,"    ,info_loss is",info_loss,"   ,total_loss",total_loss)
                #print(std.log().max(),std.log().min())
                izy_bound = math.log(10,2) - class_loss
                izx_bound = info_loss

                self.optim.zero_grad()
                total_loss.backward()
                self.optim.step()
                self.toynet_ema.update(self.toynet.state_dict())
                ## prediction over a mini-batch
                #total_num += label.size(0)
                prediction = F.softmax(logit,dim=1).max(1)[1]
                y_real=torch.cat([y_real,y],dim=0)
                y_hat=torch.cat([y_hat,prediction],dim=0)
                correct += torch.eq(prediction,y).float().sum()
                
            
                '''
                if self.num_avg != 0 :
                    _, avg_soft_logit = self.toynet(x,self.num_avg)
                    avg_prediction = avg_soft_logit.max(1)[1]
                    avg_accuracy = torch.eq(avg_prediction,y).float().mean()
                else: avg_accuracy = Variable(cuda(torch.zeros(accuracy.size()), self.cuda))
                '''
            
            if self.tensorboard :
                        self.tf.add_scalars(main_tag='performance/accuracy',
                                            tag_scalar_dict={
                                                'train_one-shot':accuracy.item(),
                                                'train_multi-shot':avg_accuracy.item()},
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/error',
                                            tag_scalar_dict={
                                                'train_one-shot':1-accuracy.item(),
                                                'train_multi-shot':1-avg_accuracy.item()},
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/cost',
                                            tag_scalar_dict={
                                                'train_one-shot_class':class_loss.item(),
                                                'train_one-shot_info':info_loss.item(),
                                                'train_one-shot_total':total_loss.item()},
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='mutual_information/train',
                                            tag_scalar_dict={
                                                'I(Z;Y)':izy_bound.item(),
                                                'I(Z;X)':izx_bound.item()},
                                            global_step=self.global_iter)
 
            
            accuracy = sklearn.metrics.accuracy_score(y_real,y_hat)
            f1score = sklearn.metrics.f1_score(y_real, y_hat,labels=None,pos_label=1, average='macro',sample_weight=None)
            print('accuracy',accuracy)
            print('f1score',f1score)
            accum_accuracy+=accuracy
            avg_accuracy= accum_accuracy/self.global_epoch
            

            #if (self.global_epoch % 2) == 0 : self.scheduler.step()
            #input accuracy and f1-score of train dataset into the dictionary
            vib_train["Accuracy"].append(float("{:.2f}".format(accuracy.item())))
            vib_train["F1_Score"].append(float("{:.2f}".format(f1score.item())))
            print(vib_train)
            
            print('[TRAIN RESULT]')
            print('i:{} IZY:{:.2f} IZX:{:.2f}'
                  .format(self.global_epoch, izy_bound.item(), izx_bound.item()), end=' ')
            print('acc:{:.4f} avg_acc:{:.4f}'
                  .format(accuracy.item(), avg_accuracy.item()), end=' ')
            print('err:{:.4f} avg_err:{:.4f}'
                  .format(1-accuracy.item(), 1-avg_accuracy.item()))
            print(classification_report(y_real,y_hat))
            
            ## valuate at each epoch
            temp_accuracy,temp_f1score=self.validate()
            
            #input accuracy and f1-score of validation dataset into another dictionary
            
            vib_valid["F1_Score"].append(float("{:.2f}".format(temp_f1score)))
            vib_valid["Accuracy"].append(float("{:.2f}".format(temp_accuracy)))
        
        print('vib_train:',vib_train)
        print('vib_validation',vib_valid)
        working_dir_path = pathlib.Path().absolute()
        SAVE_DIR_PATH = str(working_dir_path) + '/Dictionaries/VIB'
        fileName1 ='vib_train'+str(self.train_dataset_percentage)
        fileName2 ='vib_valid'+str(self.train_dataset_percentage)
        self.writeToJSONFile(SAVE_DIR_PATH,fileName1,vib_train)
        self.writeToJSONFile(SAVE_DIR_PATH,fileName2,vib_valid)
        
        print(" [*] Training Finished!")
    
    def validate(self, save_ckpt=True):
        self.set_mode('eval')
        #print(save_ckpt)
        class_loss = 0
        info_loss = 0
        total_loss = 0
        izy_bound = 0
        izx_bound = 0
        correct = 0
        avg_correct = 0
        total_num = 0
        counter=0
        accum_accuracy =0
        
       
        y_real=torch.randn([0])
        y_hat=torch.randn([0])
        ## loop over mini-batches
        for idx, (images,labels) in enumerate(self.data_loader['validate']):
            counter=counter+1
            x = Variable(cuda(images, self.cuda))
            y = Variable(cuda(labels, self.cuda))
            (mu, std), logit = self.toynet_ema.model(x)

            class_loss += F.cross_entropy(logit,y,size_average=False).div(math.log(2))
            info_loss += -0.5*(1+2*std.log()-mu.pow(2)-std.pow(2)).sum().div(math.log(2))
            total_loss += class_loss + self.beta*info_loss
            total_num += y.size(0)
            

            izy_bound += math.log(10,2) - class_loss
            izx_bound += info_loss
            prediction = F.softmax(logit,dim=1).max(1)[1]
            y_real=torch.cat([y_real,y],dim=0)
            y_hat=torch.cat([y_hat,prediction],dim=0)
            correct += torch.eq(prediction,y).float().sum()
           
 
            

        accuracy = sklearn.metrics.accuracy_score(y_real,y_hat)
        f1score = sklearn.metrics.f1_score(y_real, y_hat,labels=None,pos_label=1, average='macro',sample_weight=None)
        accum_accuracy+=accuracy
        avg_accuracy= accum_accuracy/self.global_epoch
        #print("validate accuracy",accuracy)
        #print("f1 score", f1score)
      

        izy_bound /= total_num
        izx_bound /= total_num
        class_loss /= total_num
        info_loss /= total_num
        total_loss /= total_num

        print('[Validation RESULT]')
        print('epoch:{} IZY:{:.2f} IZX:{:.2f}'
                .format(self.global_epoch, izy_bound.item(), izx_bound.item()), end=' ')

        print('acc:{:.4f} avg_acc:{:.4f}'
                .format(accuracy.item(), avg_accuracy.item()), end=' ')
        print('err:{:.4f} avg_erra:{:.4f}'
                .format(1-accuracy.item(), 1-avg_accuracy.item()))
        print(classification_report(y_real,y_hat))
       
              
        
        if self.history['f1_score'] <f1score.item():
            print('update new params')
            #print(self.history['f1_score'])
            #print(self.history['total_loss'])
            #print(self.history['epoch'])
            self.history['avg_acc'] = avg_accuracy.item()
            self.history['f1_score'] = f1score.item()
            self.history['class_loss'] = class_loss.item()
            self.history['info_loss'] = info_loss.item()
            self.history['total_loss'] = total_loss.item()
            self.history['epoch'] = self.global_epoch
            self.history['iter'] = self.global_iter
            if (save_ckpt) :
                {self.save_checkpoint('best_f1score.tar'),
                 print("save checkpoint")}
        '''
        if self.history['avg_acc'] < avg_accuracy.item() :
            print('enter update new params')
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
        
        if self.tensorboard :
            self.tf.add_scalars(main_tag='performance/accuracy',
                                tag_scalar_dict={
                                    'test_one-shot':accuracy.item(),
                                    'test_multi-shot':avg_accuracy.item()},
                                global_step=self.global_iter)
            self.tf.add_scalars(main_tag='performance/error',
                                tag_scalar_dict={
                                    'test_one-shot':1-accuracy.item(),
                                    'test_multi-shot':1-avg_accuracy.item()},
                                global_step=self.global_iter)
            self.tf.add_scalars(main_tag='performance/cost',
                                tag_scalar_dict={
                                    'test_one-shot_class':class_loss.item(),
                                    'test_one-shot_info':info_loss.item(),
                                    'test_one-shot_total':total_loss.item()},
                                global_step=self.global_iter)
            self.tf.add_scalars(main_tag='mutual_information/test',
                                tag_scalar_dict={
                                    'I(Z;Y)':izy_bound.item(),
                                    'I(Z;X)':izx_bound.item()},
                                global_step=self.global_iter)
         
       
        #if (self.global_epoch % 2) == 0 : self.scheduler.step()
        #input accuracy and f1-score of train dataset into the dictionary
        #vib_train["Accuracy"].append(float("{:.2f}".format(accuracy.item())))
        #vib_train["F1_Score"].append(float("{:.2f}".format(f1score.item())))
            
           
        #with open('vib_valid.json', 'w') as jf:
            #json.dump(vib_valid, jf)
        self.set_mode('train')
        return accuracy.item(),f1score.item()
    
    def test(self, save_ckpt=False, data_set='test'):
        """
        evaluate the performance over a specific dataset
        
        :params[in]: data_set, str, which section of data to use,
                     data_set = 'train' to use training data, 'valid' for validation data, 
                     'test' for testing dataset
                     
        
        """
        self.set_mode('eval')
        class_loss = 0
        info_loss = 0
        total_loss = 0
        izy_bound = 0
        izx_bound = 0
        correct = 0
        accum_accuracy=0
        #avg_correct = 0
        total_num = 0
        ## define empty tensors
        y_real=torch.randn([0])
        y_hat=torch.randn([0])
        ## for testing over best model
        if data_set == 'test':
            self.load_checkpoint(filename='best_f1score.tar')
        ## loop over mini-batches
        for idx, (images,labels) in enumerate(self.data_loader[data_set]):
            x = Variable(cuda(images, self.cuda))
            y = Variable(cuda(labels, self.cuda))
            (mu, std), logit = self.toynet_ema.model(x)

            class_loss += F.cross_entropy(logit,y,size_average=False).div(math.log(2))
            info_loss += -0.5*(1+2*std.log()-mu.pow(2)-std.pow(2)).sum().div(math.log(2))
            total_loss += class_loss + self.beta*info_loss
            total_num += y.size(0)
            

            izy_bound += math.log(10,2) - class_loss
            izx_bound += info_loss
            prediction = F.softmax(logit,dim=1).max(1)[1]
            y_real=torch.cat([y_real,y],dim=0)
            y_hat=torch.cat([y_hat, prediction],dim=0)
            correct += torch.eq(prediction,y).float().sum()
        accuracy = sklearn.metrics.accuracy_score(y_real,y_hat)
        f1score = sklearn.metrics.f1_score(y_real, y_hat,labels=None, average='macro',sample_weight=None)
        accum_accuracy+=accuracy
        avg_accuracy = accum_accuracy 
        
       

        izy_bound /= total_num
        izx_bound /= total_num
        class_loss /= total_num
        info_loss /= total_num
        total_loss /= total_num
        
        print('[TEST RESULT]')
        print('epoch:{} IZY:{:.2f} IZX:{:.2f}'
                .format(self.global_epoch, izy_bound.item(), izx_bound.item()), end=' ')

        print('acc:{:.4f} avg_acc:{:.4f}'
                .format(accuracy.item(), avg_accuracy.item()), end=' ')
        print('err:{:.4f} avg_erra:{:.4f}'
                .format(1-accuracy.item(), 1-avg_accuracy.item()))
        print(classification_report(y_real,y_hat))
        #print()


        if self.tensorboard :
            self.tf.add_scalars(main_tag='performance/accuracy',
                                tag_scalar_dict={
                                    'test_one-shot':accuracy.item(),
                                    'test_multi-shot':avg_accuracy.item()},
                                global_step=self.global_iter)
            self.tf.add_scalars(main_tag='performance/error',
                                tag_scalar_dict={
                                    'test_one-shot':1-accuracy.item(),
                                    'test_multi-shot':1-avg_accuracy.item()},
                                global_step=self.global_iter)
            self.tf.add_scalars(main_tag='performance/cost',
                                tag_scalar_dict={
                                    'test_one-shot_class':class_loss.item(),
                                    'test_one-shot_info':info_loss.item(),
                                    'test_one-shot_total':total_loss.item()},
                                global_step=self.global_iter)
            self.tf.add_scalars(main_tag='mutual_information/test',
                                tag_scalar_dict={
                                    'I(Z;Y)':izy_bound.item(),
                                    'I(Z;X)':izx_bound.item()},
                                global_step=self.global_iter)


    def save_checkpoint(self, filename='best_f1score.tar'):
        model_states = {
                'net':self.toynet.state_dict(),
                'net_ema':self.toynet_ema.model.state_dict(),
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
        print("=> saved checkpoint '{}' (epoch {})".format(file_path,self.global_epoch))

    def load_checkpoint(self, filename='best_f1score.tar'):
        file_path = self.ckpt_dir.joinpath(filename)
        #print(file_path)
        if file_path.is_file():
            print("=> loading checkpoint '{}'".format(file_path))
            checkpoint = torch.load(file_path.open('rb'))
            self.global_epoch = checkpoint['epoch']
            self.global_iter = checkpoint['iter']
            self.history = checkpoint['history']
            

            self.toynet.load_state_dict(checkpoint['model_states']['net'])
            #self.toynet_ema.model.load_state_dict(checkpoint['model_states']['net_ema'])

            print("=> loaded checkpoint '{} (epoch {})'".format(
                file_path, self.global_epoch))

        else:
            print("=> no checkpoint found at '{}'".format(file_path))
            
            
    def writeToJSONFile(self,path, fileName, data):
        filePathNameWExt =  path + '/' + fileName + '.json'
        with open(filePathNameWExt, 'w') as fp:
            json.dump(data, fp)
import yaml
import argparse
import pandas as pd
import csv
import os
import pandas as pd
from Feature_extract import feature_transform
from Datagenerator import Datagen
import numpy as np
import torch
from torch.utils.data import DataLoader
from Model import ProtoNet,ResNet
from tqdm import tqdm
from collections import Counter
from batch_sampler import EpisodicBatchSampler
from torch.nn import functional as F
from util import prototypical_loss as loss_fn
from util import evaluate_prototypes
from glob import glob
import hydra
from omegaconf import DictConfig, OmegaConf
import h5py




def init_seed():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)


def train_protonet(encoder,train_loader,valid_loader,conf,num_batches_tr,num_batches_vd):

    '''Model training
    Args:
    -model: Model
    -train_laoder: Training loader
    -valid_load: Valid loader
    -conf: configuration object
    -num_batches_tr: number of training batches
    -num_batches_vd: Number of validation batches
    Out:
    -best_val_acc: Best validation accuracy
    -model
    -best_state: State dictionary for the best validation accuracy
    '''

    if conf.train.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    
    optim = torch.optim.Adam([{'params':encoder.parameters()}] ,lr=conf.train.lr_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, gamma=conf.train.scheduler_gamma,
                                                   step_size=conf.train.scheduler_step_size)
    num_epochs = conf.train.epochs

    best_model_path = conf.path.best_model
    last_model_path = conf.path.last_model
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_val_acc = 0.0
    encoder.to(device)
    

    for epoch in range(num_epochs):

        print("Epoch {}".format(epoch))
        train_iterator = iter(train_loader)
        for batch in tqdm(train_iterator):
            optim.zero_grad()
            encoder.train()
            
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            x_out = encoder(x)
            tr_loss,tr_acc = loss_fn(x_out,y,conf.train.n_shot)
            train_loss.append(tr_loss.item())
            train_acc.append(tr_acc.item())

            tr_loss.backward()
            optim.step()

        avg_loss_tr = np.mean(train_loss[-num_batches_tr:])
        avg_acc_tr = np.mean(train_acc[-num_batches_tr:])
        print('Average train loss: {}  Average training accuracy: {}'.format(avg_loss_tr,avg_acc_tr))
        lr_scheduler.step()
        encoder.eval()
        
        val_iterator = iter(valid_loader)

        for batch in tqdm(val_iterator):
            x,y = batch
            x = x.to(device)
            x_val = encoder(x)
            valid_loss, valid_acc = loss_fn(x_val, y, conf.train.n_shot)
            val_loss.append(valid_loss.item())
            val_acc.append(valid_acc.item())
        avg_loss_vd = np.mean(val_loss[-num_batches_vd:])
        avg_acc_vd = np.mean(val_acc[-num_batches_vd:])

        print ('Epoch {}, Validation loss {:.4f}, Validation accuracy {:.4f}'.format(epoch,avg_loss_vd,avg_acc_vd))
        if avg_acc_vd > best_val_acc:
            print("Saving the best model with valdation accuracy {}".format(avg_acc_vd))
            best_val_acc = avg_acc_vd
            #best_state = model.state_dict()
            torch.save({'encoder':encoder.state_dict()},best_model_path)
    torch.save({'encoder':encoder.state_dict()},last_model_path)

    return best_val_acc,encoder




@hydra.main(config_name="config")
def main(conf : DictConfig):

    if not os.path.isdir(conf.path.feat_path):
        os.makedirs(conf.path.feat_path)

    if not os.path.isdir(conf.path.feat_train):
        os.makedirs(conf.path.feat_train)

    if not os.path.isdir(conf.path.feat_eval):
        os.makedirs(conf.path.feat_eval)

    if conf.set.features:

        print(" --Feature Extraction Stage--")
        Num_extract_train,data_shape = feature_transform(conf=conf,mode="train")
        print("Shape of dataset is {}".format(data_shape))
        print("Total training samples is {}".format(Num_extract_train))

        Num_extract_eval = feature_transform(conf=conf,mode='eval')
        print("Total number of samples used for evaluation: {}".format(Num_extract_eval))
        print(" --Feature Extraction Complete--")



    if conf.set.train:

        if not os.path.isdir(conf.path.Model):
            os.makedirs(conf.path.Model)

        init_seed()


        gen_train = Datagen(conf)
        X_train,Y_train,X_val,Y_val = gen_train.generate_train()
        X_tr = torch.tensor(X_train)
        Y_tr = torch.LongTensor(Y_train)
        X_val = torch.tensor(X_val)
        Y_val = torch.LongTensor(Y_val)

        samples_per_cls =  conf.train.n_shot * 2

        batch_size_tr = samples_per_cls * conf.train.k_way
        batch_size_vd = batch_size_tr
        
        if conf.train.num_episodes is not None:

            num_episodes_tr = conf.train.num_episodes
            num_episodes_vd = conf.train.num_episodes

        else:

            num_episodes_tr = len(Y_train)//batch_size_tr
            num_episodes_vd = len(Y_val)//batch_size_vd

        
        
        
        

        samplr_train = EpisodicBatchSampler(Y_train,num_episodes_tr,conf.train.k_way,samples_per_cls)
        samplr_valid = EpisodicBatchSampler(Y_val,num_episodes_vd,conf.train.k_way,samples_per_cls)

        train_dataset = torch.utils.data.TensorDataset(X_tr,Y_tr)
        valid_dataset = torch.utils.data.TensorDataset(X_val,Y_val)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_sampler=samplr_train,num_workers=0,pin_memory=True,shuffle=False)
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,batch_sampler=samplr_valid,num_workers=0,pin_memory=True,shuffle=False)

        if conf.train.encoder == 'Resnet':
            encoder  = ResNet()
        else:
            encoder = ProtoNet()

        
        best_acc,model,best_state = train_protonet(encoder,train_loader,valid_loader,conf,num_episodes_tr,num_episodes_vd)
        print("Best accuracy of the model on training set is {}".format(best_acc))

    if conf.set.eval:

        device = 'cuda'

        init_seed()


        name_arr = np.array([])
        onset_arr = np.array([])
        offset_arr = np.array([])
        all_feat_files = [file for file in glob(os.path.join(conf.path.feat_eval,'*.h5'))]

        for feat_file in all_feat_files:
            feat_name = feat_file.split('/')[-1]
            audio_name = feat_name.replace('h5','wav')

            print("Processing audio file : {}".format(audio_name))

            hdf_eval = h5py.File(feat_file,'r')
            strt_index_query =  hdf_eval['start_index_query'][:][0]
            
            
            onset,offset = evaluate_prototypes(conf,hdf_eval,device,strt_index_query)

            name = np.repeat(audio_name,len(onset))
            name_arr = np.append(name_arr,name)
            onset_arr = np.append(onset_arr,onset)
            offset_arr = np.append(offset_arr,offset)

        df_out = pd.DataFrame({'Audiofilename':name_arr,'Starttime':onset_arr,'Endtime':offset_arr})
        csv_path = os.path.join(conf.path.root_dir,'Eval_out.csv')
        df_out.to_csv(csv_path,index=False)


if __name__ == '__main__':
     main()


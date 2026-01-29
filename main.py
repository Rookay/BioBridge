# -*- coding: UTF-8 -*-
# @Project ：BioEEG-VisDec 
# @File    ：main.py
# @IDE     ：PyCharm 
# @Date    ：2025/11/21 13:58
from omegaconf import OmegaConf
from torch import optim

from dataset import load_eeg_data
from model import Simple_Model
from trainer import Trainer

if __name__ == '__main__':
    config = OmegaConf.load("config.yaml")
    config['data']['subjects'] = ["sub-07"]
    train_loader,val_loader, test_loader = load_eeg_data(config)
    model = Simple_Model(config)
    optimizer = optim.Adam(model.parameters(), lr=config['train']['lr'])
    trainer = Trainer(model, train_loader, val_loader, test_loader, optimizer,config)

    trainer.train()



# -*- coding: UTF-8 -*-
# @Project ：BioEEG-VisDec 
# @File    ：dataset.py
# @IDE     ：PyCharm 
# @Date    ：2025/11/19 14:23
import argparse

import torch, os
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging

from torch.utils.data import DataLoader
from torchvision import transforms


def load_eeg_data(config):
    exp_setting = config.get('exp_setting', 'intra-subject')
    print(exp_setting)

    if exp_setting == 'intra-subject':
        test_dataset = EEGDataset(config, mode='test')
        print('init test_dataset success')
        train_dataset = EEGDataset(config, mode='train')
        print('init train_dataset success')
        test_loader = DataLoader(test_dataset, batch_size=config['data']['test_batch_size'], shuffle=False,
                                 drop_last=False, num_workers=0, pin_memory=True)
        train_loader = DataLoader(train_dataset, batch_size=config['data']['train_batch_size'], shuffle=True,
                                  drop_last=False, num_workers=0, pin_memory=True)

        # for i in range (0,200):
        #     print(test_dataset[i]["label"])
        return train_loader,test_loader,test_loader

    elif exp_setting == 'inter-subject':
        subjects = config['data']['subjects']
        test_dataset = EEGDataset(config, mode='test')
        print('init test_dataset success')

        all_subjects = [f'sub-{i:02}' for i in range(1, 11)]
        leave_one_subjects = list(set(all_subjects) - set(subjects))
        leave_one_subjects_config = config
        leave_one_subjects_config['data']['subjects'] = leave_one_subjects
        val_dataset = EEGDataset(leave_one_subjects_config, mode='test')
        print('init val_dataset success')
        train_dataset = EEGDataset(leave_one_subjects_config, mode='train')
        print('init train_dataset success')
        test_loader = DataLoader(test_dataset, batch_size=config['data']['test_batch_size'], shuffle=False,
                                 drop_last=False, num_workers=0)  # , pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config['data']['val_batch_size'], shuffle=False,
                                drop_last=False, num_workers=0)  # , pin_memory=True)
        train_loader = DataLoader(train_dataset, batch_size=config['data']['train_batch_size'], shuffle=True,
                                  drop_last=False, num_workers=0)  # , pin_memory=True)
        return train_loader,val_loader,test_loader


class EEGDataset(Dataset):
    def __init__(self, config, mode):
        self.data_dir = config['data']['data_dir']
        self.subjects = config['data']['subjects']
        print(f'subjects:{self.subjects}')
        self.mode = mode
        self.selected_ch = config['data']['selected_ch']
        self.channels = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                         'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
                         'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                         'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
                         'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                         'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                         'O1', 'Oz', 'O2']
        if self.selected_ch == "None":
            self.selected_ch = self.channels

        self.avg = config['data'][f"{mode}_avg"]

        self.timesteps = config['data']['timesteps']

        self.n_cls = 1654 if self.mode == 'train' else 200
        self.per_trials = 4 if self.mode == 'train' else 80

        self.data_paths = [os.path.join(self.data_dir, subject, f'{mode}.pt') for subject in self.subjects]
        self.loaded_data = [self.load_data(data_path, config) for data_path in self.data_paths]

        self.trial_subject = self.loaded_data[0]['eeg'].shape[0]
        self.trial_all_subjects = self.trial_subject * len(self.subjects)


        self.match_label = np.ones(self.trial_all_subjects, dtype=int)

        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  
        ])

    def load_data(self, data_path, config):
        logging.info(f"----load {data_path.rsplit('1000HZ', 1)[-1]}----")
        loaded_data = torch.load(data_path, weights_only=False)
        loaded_data['eeg'] = torch.from_numpy(loaded_data['eeg'])

        if self.selected_ch:
            selected_idx = [self.channels.index(ch) for ch in self.selected_ch]
            loaded_data['eeg'] = loaded_data['eeg'][:, :, selected_idx]
        if self.avg:
            avg_data = {}
            avg_data['eeg'] = loaded_data['eeg'].mean(axis=1)
            avg_data['label'] = loaded_data['label'][:, 0]
            avg_data['img'] = loaded_data['img'][:, 0]
            avg_data['text'] = loaded_data['text'][:, 0]

            avg_data['session'] = loaded_data['session']
            avg_data['times'] = loaded_data['times']
            avg_data['img'] = np.array(avg_data['img'])
            loaded_data = avg_data
        else:
            _data = {}
            _data['eeg'] = loaded_data['eeg'].reshape(-1, *loaded_data['eeg'].shape[2:])
            _data['eeg_avg'] = loaded_data['eeg'].mean(axis=1)
            _data['label'] = loaded_data['label'].reshape(-1)
            _data['img'] = loaded_data['img'].reshape(-1)
            _data['text'] = loaded_data['text'].reshape(-1)
            _data['session'] = loaded_data['session'].reshape(-1)
            _data['times'] = loaded_data['times']
            _data['img'] = loaded_data['img'].reshape(-1)
            _data['img'] = np.array(_data['img'])

            loaded_data = _data

        for k, v in loaded_data.items():
            if k in ['eeg', 'label', 'img', 'text', 'session']:
                logging.info(f"{k}: {v.shape}")
        # print(loaded_data["img"])

        return loaded_data


    def __getitem__(self, index):

        subject = index // self.trial_subject
        trial_index = index % self.trial_subject

        eeg = self.loaded_data[subject]['eeg'][trial_index].float()
        if self.avg:
            eeg_mean = eeg
        else:
            eeg_mean = self.loaded_data[subject]['eeg_avg'][trial_index // self.per_trials].float()

        label = self.loaded_data[subject]['label'][trial_index]
        img_path = self.loaded_data[subject]['img'][trial_index]

        img = Image.open(img_path).convert("RGB")
        img = self.img_transform(img)

        # match_label = self.match_label[index]

        text = f"This is a {self.loaded_data[subject]['text'][trial_index]}."
        # text_features = self.text_features[self.loaded_data[subject]['text'][trial_index]]
        session = self.loaded_data[subject]['session'][trial_index]

        sample = {
            'idx': index,
            'eeg': eeg[:, self.timesteps[0]:self.timesteps[1]],
            'label': label,
            'img_path': img_path,
            'img': img,
            # 'img_features': img_features,
            'text': text,
            # 'text_features': text_features,
            'session': session,
            'subject': subject,
            'eeg_mean': eeg_mean[:, self.timesteps[0]:self.timesteps[1]],
        }
        return sample

    def __len__(self):
        return self.trial_all_subjects


if __name__ == '__main__':

    config = OmegaConf.load("config.yaml")
    config['data']['subjects'] = ["sub-01"]

    train_loader, test_loader = load_eeg_data(config)







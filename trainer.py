# -*- coding: UTF-8 -*-
# @Project ：BioEEG-VisDec
# @File    ：trainer.py
# @IDE     ：PyCharm
# @Date    ：2025/11/21 13:53
import csv
from datetime import datetime

import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

def cosine_sim(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return a @ b.t()

def zero_shot_accuracy(sim_matrix, labels):
    # print(sim_matrix.shape)
    ranks = sim_matrix.argsort(dim=1, descending=True)
    top1 = (ranks[:, 0] == labels).float().mean().item()
    top5 = (ranks[:, :5] == labels[:, None]).any(dim=1)
    sum_top5 = 0
    for i in top5:
        if i:
            sum_top5+=1


    return top1, sum_top5/len(labels)

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader,
                 optimizer, config = None, device="cuda"):
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.exp_setting = config.get('exp_setting', 'intra-subject')
        self.alpha = 0.5 if self.exp_setting == 'intra-subject' else 0.9
        print(self.exp_setting)
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device

    def train_step(self, batch):
        self.model.train()
        eeg = batch['eeg'].to(self.device)
        img = batch['img'].to(self.device)

        low_eeg_feat, low_img_feat, high_eeg_feat, high_img_feat = self.model(eeg, img)
        logit_scale = self.model.low_eeg_encoder.logit_scale.exp()
        low_loss = self.clip_loss(low_eeg_feat, low_img_feat, logit_scale)
        high_loss = self.clip_loss(high_eeg_feat, high_img_feat, logit_scale)
        loss = self.alpha * low_loss + (1 - self.alpha) * high_loss


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        del eeg, img, low_eeg_feat, low_img_feat, high_eeg_feat, high_img_feat

        return loss.item()

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total_loss, total_top1, total_top5, total_num = 0, 0, 0, 0
        total_low_top1,total_low_top5,total_high_top1,total_high_top5 = 0, 0, 0, 0
        for batch in tqdm(loader, desc="Validation"):
            eeg = batch['eeg'].to(self.device)
            img = batch['img'].to(self.device)
            labels = batch['label'].to(self.device)

            low_eeg_feat, low_img_feat, high_eeg_feat, high_img_feat = self.model(eeg, img)
            logit_scale = self.model.low_eeg_encoder.logit_scale.exp()
            low_loss = self.clip_loss(low_eeg_feat, low_img_feat, logit_scale)
            high_loss = self.clip_loss(high_eeg_feat, high_img_feat, logit_scale)
            loss = self.alpha * low_loss +(1-self.alpha)* high_loss
            batch_size = eeg.size(0)
            total_num += batch_size

            low_sim = cosine_sim(low_eeg_feat, low_img_feat)
            high_sim = cosine_sim(high_eeg_feat, high_img_feat)
            sim = torch.max(low_sim, high_sim)
            low_top1, low_top5 = zero_shot_accuracy(low_sim, labels)
            high_top1, high_top5 = zero_shot_accuracy(high_sim, labels)
            top1, top5 = zero_shot_accuracy(sim, labels)

            total_low_top1 += low_top1 * batch_size
            total_low_top5 += low_top5 * batch_size
            total_high_top1 += high_top1 * batch_size
            total_high_top5 += high_top5 * batch_size
            total_top1 += top1 * batch_size
            total_top5 += top5 * batch_size

            total_loss += loss.item() * batch_size


        avg_loss = total_loss / total_num
        avg_low_top1 = total_low_top1 / total_num
        avg_low_top5 = total_low_top5 / total_num
        avg_high_top1 = total_high_top1 / total_num
        avg_high_top5 = total_high_top5 / total_num
        avg_top1 = total_top1 / total_num
        avg_top5 = total_top5 / total_num
        del eeg, img, labels
        del low_eeg_feat, low_img_feat, high_eeg_feat, high_img_feat
        del low_sim, high_sim, sim

        return avg_loss,avg_low_top1,avg_low_top5,avg_high_top1,avg_high_top5,avg_top1,avg_top5


    def clip_loss(self, eeg_feat, img_feat, logit_scale):
        eeg_feat = F.normalize(eeg_feat, dim=-1)
        img_feat = F.normalize(img_feat, dim=-1)
        logits = logit_scale * eeg_feat @ img_feat.t()
        labels = torch.arange(len(logits), dtype=torch.long, device=logits.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        return (loss_i + loss_t) / 2


    def save(self, model_path, name="model.pt"):
        torch.save(self.model.state_dict(), os.path.join(model_path, name))

    def load(self, path="model.pt"):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def train(self):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join("models",self.exp_setting, "_".join(self.config['data']['subjects']))
        model_path = os.path.join(model_dir, f"model_{timestamp}")
        os.makedirs(model_path, exist_ok=True)  # 注意要创建目录
        log_file = os.path.join(model_path, f"log.csv")

        with open(log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch",
                             "train_loss",
                             "test_loss",
                             "test_accuracy",
                             "top5_acc"])


        best_val_loss = 100
        best_val_top1 = 0.0
        best_val_top5 = 0.0
        best_epoch = 0
        for epoch in range(1, self.config['train']['epoch'] + 1):
            losses, top1s, top5s = [], [], []
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
                loss = self.train_step(batch)
                losses.append(loss)

            avg_loss = sum(losses) / len(losses)

            val_loss, avg_low_top1, avg_low_top5, avg_high_top1, avg_high_top5, avg_top1, avg_top5 = self.validate(self.val_loader)
            print(f"Epoch {epoch} | Train Loss {avg_loss:.4f}  "
                  f"| Val Loss {val_loss:.4f} | Val Top1 {avg_top1:.4f} | Val Top5 {avg_top5:.4f} "
                  f"|Val low Top1 {avg_low_top1:.4f} |Val low Top5 {avg_low_top5}"
                  f"|Val high Top1 {avg_high_top1:.4f} |Val high Top5 {avg_high_top5}")
            if self.exp_setting == 'inter-subject':
                test_loss, test_avg_low_top1, test_avg_low_top5, test_avg_high_top1, test_avg_high_top5, test_avg_top1, test_avg_top5 = self.validate(self.test_loader)
                print(test_avg_top1,test_avg_top5)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_top1 = avg_top1
                best_val_top5 = avg_top5
                best_epoch = epoch
                self.save(model_path, f"best_model.pt")

            # if epoch % 5 == 0:
            #     self.save(model_path, f"epoch_{epoch}.pt")

            with open(log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    avg_loss,
                    val_loss,
                    avg_top1,
                    avg_top5
                ])



        print(f"best val loss: {best_val_loss:.4f},best val top1: {best_val_top1:.4f}, best val top5: {best_val_top5:.4f}, best epoch: {best_epoch}")


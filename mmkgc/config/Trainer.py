              
from calendar import c
import logging
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
import copy
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

logger = logging.getLogger(__name__)


class Trainer(object):

    def __init__(self,
                 model=None,
                 data_loader=None,
                 train_times=1000,
                 alpha=0.5,
                 use_gpu=True,
                 opt_method="sgd",
                 save_steps=None,
                 checkpoint_dir=None,
                 weight_decay=0.0,
                 mu=None,
                 tester=None,
                 test_interval=100,
                 early_stop_delta=0.01,
                 early_stop_patience=100,
                 metric_name="hit1",
                 metric_delta=0.0001,
                 metric_patience=5,
                 lr_decay_patience=3,
                 lr_decay_factor=0.5):

        self.work_threads = 8
        self.train_times = train_times

        self.opt_method = opt_method
        self.optimizer = None
        self.lr_decay = 0
        self.weight_decay = weight_decay
        self.alpha = alpha
                                        

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu
        self.save_steps = save_steps
        self.checkpoint_dir = checkpoint_dir
        self.scaler = GradScaler(enabled=self.use_gpu)
        self.tester = tester
        self.test_interval = test_interval
        self.early_stop_delta = early_stop_delta
        self.early_stop_patience = early_stop_patience
        self.loss_history = []
        self.metric_name = metric_name
        self.metric_delta = metric_delta
        self.metric_patience = metric_patience
        self.lr_decay_patience = lr_decay_patience
        self.lr_decay_factor = lr_decay_factor
        self.best_metric = None
        self.bad_eval_count = 0
        self.bad_lr_count = 0


    def train_one_step(self, data):

        self.optimizer.zero_grad()
        with autocast(enabled=self.use_gpu):
            loss = self.model({
                'batch_h': self.to_var(data['batch_h'], self.use_gpu),
                'batch_t': self.to_var(data['batch_t'], self.use_gpu),
                'batch_r': self.to_var(data['batch_r'], self.use_gpu),
                'batch_y': self.to_var(data['batch_y'], self.use_gpu),
                'mode': data['mode']
            })
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()

    def run(self):
        if self.use_gpu:
            self.model.cuda()

        if self.optimizer is not None:
            pass
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        else:
            raise NotImplementedError
        print("Finish initializing...")

        training_range = tqdm(range(self.train_times))
        for epoch in training_range:
            res = 0.0
            for data in self.data_loader:
                loss = self.train_one_step(data)
                res += loss
            training_range.set_description("Epoch %d | D loss: %f" % (epoch, res))

                                                       
            if self.early_stop_patience and self.early_stop_patience > 0:
                self.loss_history.append(res)
                if len(self.loss_history) >= self.early_stop_patience:
                    prev = self.loss_history[-self.early_stop_patience]
                    improvement = prev - self.loss_history[-1]
                    if improvement < self.early_stop_delta:
                        print("Early stopping: loss improvement %.6f < %.6f in last %d epochs" % (
                            improvement, self.early_stop_delta, self.early_stop_patience))
                        break

                                 
            if self.tester is not None and (epoch + 1) % self.test_interval == 0:
                print("Running evaluation at epoch %d" % (epoch + 1))
                try:
                    mrr, mr, hit10, hit3, hit1 = self.tester.run_link_prediction(type_constrain=False)
                    metrics = {
                        "mrr": mrr,
                        "mr": mr,
                        "hit10": hit10,
                        "hit3": hit3,
                        "hit1": hit1,
                    }
                    log_msg = (
                        "Eval | Epoch %d | MRR: %.4f Hit@10: %.4f Hit@1: %.4f" %
                        (epoch + 1, mrr, hit10, hit1)
                    )
                    logger.info(log_msg)
                    print(log_msg)
                    current = metrics.get(self.metric_name, hit1)
                    if self.best_metric is None or current > self.best_metric + self.metric_delta:
                        self.best_metric = current
                        self.bad_eval_count = 0
                        self.bad_lr_count = 0
                    else:
                        self.bad_eval_count += 1
                        self.bad_lr_count += 1
                        if self.lr_decay_patience and self.bad_lr_count >= self.lr_decay_patience:
                            for pg in self.optimizer.param_groups:
                                pg["lr"] = pg["lr"] * self.lr_decay_factor
                            decay_msg = (
                                "LR decayed to %.6g due to no %s improvement" % (
                                    self.optimizer.param_groups[0]["lr"], self.metric_name)
                            )
                            logger.info(decay_msg)
                            print(decay_msg)
                            self.bad_lr_count = 0
                        if self.metric_patience and self.bad_eval_count >= self.metric_patience:
                            stop_msg = (
                                "Early stopping: %s did not improve for %d evals" % (
                                    self.metric_name, self.metric_patience)
                            )
                            logger.info(stop_msg)
                            print(stop_msg)
                            break
                except Exception as e:
                    print("Evaluation failed:", e)

            if self.save_steps and self.checkpoint_dir and (epoch + 1) % self.save_steps == 0:
                print("Epoch %d has finished, saving..." % (epoch))
                base_model = getattr(self.model, "model", self.model)
                base_model.save_checkpoint(os.path.join(self.checkpoint_dir + "-" + str(epoch) + ".ckpt"))

    def set_model(self, model):
        self.model = model

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_train_times(self, train_times):
        self.train_times = train_times

    def set_save_steps(self, save_steps, checkpoint_dir=None):
        self.save_steps = save_steps
        if not self.checkpoint_dir:
            self.set_checkpoint_dir(checkpoint_dir)

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir







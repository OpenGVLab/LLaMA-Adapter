# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import json
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
import os
from typing import Iterable
import contextlib

import torch

import util.misc as misc
import util.lr_sched as lr_sched
import pdb

def train_one_epoch(model: torch.nn.Module,
                    data_loader, optimizer: torch.optim.Optimizer,
                    epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, (examples, labels, example_mask, imgs) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)):

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate_epoch(optimizer, data_iter_step / len(data_loader) + epoch, args)

        autocast_ctx = {
            "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
            "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
            "tf32": contextlib.nullcontext(),
        }[args.precision]
        with autocast_ctx:
             c_loss, m_loss, _, _ = model(examples, labels)
        loss = c_loss  + m_loss * 0
        loss_value = loss.item()
        c_loss_value = c_loss.item()
        m_loss_value = m_loss
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        update_grad = (data_iter_step + 1) % accum_iter == 0
        grad_norm = loss_scaler(
            loss, optimizer, model,
            parameters=model.parameters(),
            update_grad=update_grad,
            clip_grad=None if args.clip_grad <= 0 else args.clip_grad,
        )
        if update_grad:
            assert grad_norm is not None
            metric_logger.update(grad_norm=grad_norm)

            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(closs=c_loss_value)
        metric_logger.update(mloss=m_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)
        m_loss_value_reduce = misc.all_reduce_mean(m_loss_value)
        if update_grad:
            grad_norm_reduce = misc.all_reduce_mean(grad_norm)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('c_train_loss', c_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('m_train_loss', m_loss_value_reduce, epoch_1000x)
            if update_grad:
                log_writer.add_scalar('grad_norm', grad_norm_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@ torch.no_grad()
def val_one_epoch(model: torch.nn.Module,
                  data_loader: Iterable, epoch: int,
                  log_writer=None,
                  args=None):
    print("!!!start validation!!!")
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, (examples, labels, example_mask) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
             c_loss, m_loss, _, _ = model(examples, labels)
        loss = c_loss  + m_loss * 0
        loss_value = loss.item()
        c_loss_value = c_loss.item()
        m_loss_value = m_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(closs=c_loss_value)
        metric_logger.update(mloss=m_loss_value)


        # loss_value_reduce = misc.all_reduce_mean(loss_value)
        # c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)
        # m_loss_value_reduce = misc.all_reduce_mean(m_loss_value)
        # if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
        #     """ We use epoch_1000x as the x-axis in tensorboard.
        #     This calibrates different curves when batch size changes.
        #     """
        #     epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
        #     log_writer.add_scalar('c_train_loss', c_loss_value_reduce, epoch_1000x)
        #     log_writer.add_scalar('m_train_loss', m_loss_value_reduce, epoch_1000x)
        #     log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


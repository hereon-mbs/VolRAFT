import os
import argparse

import torch
import torch.nn as nn

import numpy as np
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity

from models.model import Model
from models.torchsummary import summary
from models.model_module import *

from utils import *
from models.VolRAFT import *
from models.VolRAFT.volraft import ModuleVolRAFT

# Convolutional Neural Network (base class)
class NetworkVolRAFT(nn.Module):
    def __init__(self,
                 patch_shape,
                 flow_shape,
                 config,
                 dtype = torch.float32):
        super(NetworkVolRAFT, self).__init__()

        # Define a name for this network
        self.name = config["network_name"]

        # Define the data type
        self.dtype = dtype

        # Define patch shape
        self.patch_nb, self.patch_nc, self.patch_nx, self.patch_ny, self.patch_nz = patch_shape

        print(f'network patch_shape = {patch_shape}')

        # Define flow shape
        (self.flow_nb, self.flow_nc, self.flow_nx, self.flow_ny, self.flow_nz) = flow_shape

        print(f'network flow_shape = {flow_shape}')

        # Get the maximum projection
        self.proj_max_flow = config["proj_max_flow"]

        # Store the config
        self.config = config

        # Prepare arguments
        args = argparse.Namespace()
        args.corr_levels = config["corr_levels"]
        args.corr_radius = config["corr_radius"]
        args.num_channels = 1 # number of channels of volumes
        args.small = True
        args.dropout = 0.0
        args.lr = config["lr"]
        args.wdecay = config["optimizer_weight_decay"]
        args.epsilon = config["optimizer_eps"]
        args.gpus = [0, 1]
        args.iters = config["iters"]
        args.num_steps = config["epochs"]
        args.mixed_precision = True
        args.clip = config["gradient_norm_clip"] # clip of gradient norm
        args.gamma = config["loss_gamma"] # exponential weighting

        if "should_normalize" in config:
            args.should_normalize = config["should_normalize"]
        else:
            args.should_normalize = True
        print(f'network_volraft.should_normalize = {args.should_normalize}')

        args.flow_shape = flow_shape

        self.module = ModuleVolRAFT(args)

        return
    
    # Get network summary as a string
    def print_network_summary(self, device, patch_shape) -> None:
        def count_parameters(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        print(f"Parameter Count: {count_parameters(self.module)}")

        return
    
    def initialize_weights(self, mode='zero'):
        return

    # Projection
    def projection(self, out):
        # project to certain range
        return out
    
    # Clip gradient norm
    def clip_gradient_norm(self):
        if 'gradient_norm_clip' in self.config.keys():
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), 
                                           max_norm = self.config["gradient_norm_clip"])
        
        return

    def forward(self, vol0, vol1):
        flow_predicts = self.module(vol0, vol1, iters = self.config["iters"])

        return flow_predicts

class ModelVolRAFT(Model):
    def __init__(self,
                 patch_shape,
                 flow_shape,
                 config,
                 dtype = torch.float32):
        super().__init__()

        self.network = NetworkVolRAFT(patch_shape, flow_shape, config, dtype)
        self.name = self.network.name

    def print_network_summary(self, device, patch_shape) -> None:
        self.network.print_network_summary(device = device, patch_shape = patch_shape)
        return
    
    def train_mode(self) -> None:
        if self.network is not None:
            self.network.train()
        return
    
    def eval_mode(self) -> None:
        if self.network is not None:
            self.network.eval()
        return
    
    def forward(self, vol0, vol1):
        return self.network.forward(vol0, vol1)
    
    def initialize_weights(self, mode = 'zero') -> None:
        self.network.initialize_weights(mode)

        return 
    
    def to(self, device):
        self.network = self.network.to(device)

        return self
    
    def parameters(self):
        return self.network.parameters()

    def load_state_dict_with_mapping(self, state_dict):
        new_state_dict = {}
        
        # Iterate through the original state dict
        for key, value in state_dict.items():
            # Detect if the key contains "model" and replace it with "module"
            new_key = key.replace("model", "module") if "model" in key else key
            new_state_dict[new_key] = value
        
        # Load the modified state dict into the network
        self.network.load_state_dict(new_state_dict)

        return
    
    def load_state_dict(self, state_dict):
        # print(f'-----------------------')
        # print(f'model_volraft load state dict function called')
        # print(f'state_dict: ')
        # print(state_dict)
        # print(f'-----------------------')
        # return self.network.load_state_dict(state_dict)
        self.load_state_dict_with_mapping(state_dict)

        return
    
    def save_checkpoint(self,
                        folder_path, 
                        patch_shape, 
                        flow_shape, 
                        optimizer, 
                        scheduler, 
                        epoch, 
                        loss_list_train, 
                        loss_list_valid):
        # Define the extension
        extension = 'pt'

        # Make the file path
        checkpoint_path = os.path.join(folder_path, 
                                    f'model_epoch{epoch:06d}.{extension}')
        
        # Save the checkpoint
        torch.save({
            'epoch': epoch,
            'patch_shape': patch_shape,
            'flow_shape': flow_shape,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss_list_train': loss_list_train,
            'loss_list_valid': loss_list_valid
            }, checkpoint_path)

        return
    
    def eval(self):
        """
        Evalute this model
        """
        self.network.eval()

        return
    
    def train(self,
              args, 
              config: dict, 
              epochs: int,
              logger: Logger,
              plot_manager: PlotManager,
              dataloader_train: DataLoader,
              dataloader_valid: DataLoader,
              patch_shape,
              flow_shape,
              optimizer: torch.optim.Optimizer, 
              scheduler: torch.optim.lr_scheduler, 
              criterion: nn.Module,
              device: torch.device, 
              ptdtype = torch.float32):
        """
        Train the model

        This is a supervised training method
        """

        def float_format(num):
            '''Format floating number'''
            # Use 'f' format for numbers within a certain range
            if 0.001 <= np.abs(num) < 1e6:
                return f"{num:7.4f}"
            # Use 'g' format for very small or very large numbers
            else:
                return f"{num:7.6g}"
            
        # Record loss
        loss_list_train = np.zeros((epochs, int(config["minibatch_num_train"])), dtype=np.float32)
        loss_list_train_info = np.zeros((epochs, int(config["minibatch_num_train"])), dtype=np.float32)
        loss_list_valid = np.zeros((epochs, int(config["minibatch_num_valid"])), dtype=np.float32)
        loss_list_valid_info = np.zeros((epochs, int(config["minibatch_num_valid"])), dtype=np.float32)
        lr_list = np.zeros((epochs), dtype=np.float32)

        epoch_idx = 0

        # Do a test on the dataloading
        # For each epoch
        epoch_plot = int(config["epoch_plot"])
        epoch_checkpoint = int(config["epoch_checkpoint"])
        epoch_validation = int(config["epoch_validation"])
        while epoch_idx < epochs:
            # Garbage collection
            DeviceManager.garbage()

            # Prepare for this epoch
            self.train_mode()

            # for each batchf
            for batch_idx, (patch_vol0, patch_vol1, patch_mask, patch_flow) in enumerate(dataloader_train):
                # Load tensors to device
                patch_vol0 = patch_vol0.to(device, dtype=ptdtype, non_blocking=True).requires_grad_(False)
                patch_vol1 = patch_vol1.to(device, dtype=ptdtype, non_blocking=True).requires_grad_(False)
                patch_mask = patch_mask.to(device, dtype=ptdtype, non_blocking=True).requires_grad_(False)
                patch_flow = patch_flow.to(device, dtype=ptdtype, non_blocking=True).requires_grad_(False)

                # Zero gradient
                optimizer.zero_grad()

                # Prediction
                flow_predict_train = self.forward(patch_vol0, patch_vol1)

                # Compute the loss
                loss_train, loss_train_info = criterion(predict = flow_predict_train,
                                                        target = patch_flow,
                                                        mask = patch_mask)
                
                # Back-propagation
                loss_train.backward()

                # Clip gradient norm if needed
                if 'gradient_norm_clip' in config.keys():
                    self.network.clip_gradient_norm()

                # Steps
                optimizer.step()

                # Make sure no_grad here
                with torch.no_grad():
                    # Store epoch information
                    loss_list_train[epoch_idx, batch_idx] = loss_train.item()

                    # Take the first information components as the metric
                    loss_list_train_info[epoch_idx, batch_idx] = loss_train_info[0]

                    # Show the loss
                    if args.debug:
                        if args.verbose_batch:
                            print(f'epoch {epoch_idx:05d} train batch {batch_idx:05d}: train loss = {float_format(loss_list_train[epoch_idx, batch_idx])}, epe = {float_format(loss_list_train_info[epoch_idx, batch_idx])}')

            # Scheduler stepping
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            lr_list[epoch_idx] = lr

            # Make sure no_grad here
            with torch.no_grad():
                # Calculate the average loss for training and validation
                if epoch_idx % epoch_validation == 0 or epoch_idx == epochs - 1:
                    # Evaluate the network
                    self.eval_mode()

                    # for each batch
                    for valid_idx, (patch_vol0, patch_vol1, patch_mask, patch_flow) in enumerate(dataloader_valid):
                        # Load tensors to device
                        patch_vol0 = patch_vol0.to(device, dtype=ptdtype, non_blocking=True).requires_grad_(False)
                        patch_vol1 = patch_vol1.to(device, dtype=ptdtype, non_blocking=True).requires_grad_(False)
                        patch_mask = patch_mask.to(device, dtype=ptdtype, non_blocking=True).requires_grad_(False)
                        patch_flow = patch_flow.to(device, dtype=ptdtype, non_blocking=True).requires_grad_(False)

                        # Prediction
                        flow_predict_valid = self.forward(patch_vol0, patch_vol1)

                        # Calculate the loss
                        loss_valid, loss_valid_info = criterion(predict = flow_predict_valid, 
                                                                target = patch_flow, 
                                                                mask = patch_mask)

                        # Store epoch information
                        loss_list_valid[epoch_idx, valid_idx] = loss_valid.item()

                        # Take the first information components as the metric
                        loss_list_valid_info[epoch_idx, valid_idx] = loss_valid_info[0]

                        # Show the loss
                        if args.debug:
                            if args.verbose_batch:
                                print(f'epoch {epoch_idx:05d} valid batch {valid_idx:05d}: valid loss = {float_format(loss_list_valid[epoch_idx, valid_idx])}, epe = {float_format(loss_list_valid_info[epoch_idx, valid_idx])}')

            loss_list_train_epoch = np.mean(loss_list_train, axis = 1) 
            loss_list_train_info_epoch = np.mean(loss_list_train_info, axis = 1)
            loss_list_valid_epoch = np.mean(loss_list_valid, axis = 1)
            loss_list_valid_info_epoch = np.mean(loss_list_valid_info, axis = 1)

            # Calculate the average loss for training and validation
            if epoch_idx % epoch_plot == 0 or epoch_idx == epochs - 1:
                epoch_indices = np.arange(epoch_idx + 1, dtype = int)
                plot_manager.plot_loss(epoch_indices = epoch_indices,
                                loss_train = loss_list_train_epoch[epoch_indices],
                                loss_valid = loss_list_valid_epoch[epoch_indices],
                                epoch_validation = epoch_validation)            
                logger.save_fig(filename = f'loss_{logger.time_stamp}')
                logger.save_fig(filename = f'loss_{logger.time_stamp}', format='pdf')

                plot_manager.plot_metrics(epoch_indices = epoch_indices,
                                    metrics_train = loss_list_train_info_epoch[epoch_indices],
                                    metrics_valid = loss_list_valid_info_epoch[epoch_indices],
                                    metrics_name = ['EPE', 'Endpoint Error'],
                                    epoch_validation = epoch_validation)
                logger.save_fig(filename = f'metrics_{logger.time_stamp}')
                logger.save_fig(filename = f'metrics_{logger.time_stamp}', format='pdf')

                plot_manager.plot_lr(epoch_indices = epoch_indices,
                                lr = lr_list[epoch_indices])
                logger.save_fig(filename = f'lr_{logger.time_stamp}')
                logger.save_fig(filename = f'lr_{logger.time_stamp}', format='pdf')

            # Save checkpoint
            if epoch_idx % epoch_checkpoint == 0 or epoch_idx == epochs - 1:
                self.save_checkpoint(folder_path = logger.checkpoint_folder_path,
                                     patch_shape = patch_shape,
                                     flow_shape = flow_shape,
                                     optimizer = optimizer,
                                     scheduler = scheduler,
                                     epoch = epoch_idx,
                                     loss_list_train = loss_list_train,
                                     loss_list_valid = loss_list_valid)
            
            # Write to log
            if epoch_idx % epoch_plot == 0 or epoch_idx == epochs - 1:
                print(f'epoch {epoch_idx:05d} lr {lr:.4e} mean [loss, epe]: train = [{float_format(loss_list_train_epoch[epoch_idx])}, {float_format(loss_list_train_info_epoch[epoch_idx])}], valid = [{float_format(loss_list_valid_epoch[epoch_idx])}, {float_format(loss_list_valid_info_epoch[epoch_idx])}]')
            else:
                print(f'epoch {epoch_idx:05d} lr {lr:.4e} mean [loss, epe]: train = [{float_format(loss_list_train_epoch[epoch_idx])}, {float_format(loss_list_train_info_epoch[epoch_idx])}]')

            # Finished for a batch
            epoch_idx += 1
        
        return
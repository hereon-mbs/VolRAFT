import os

import torch
import torch.nn as nn

import numpy as np
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity

from models.model import Model
from models.torchsummary import summary
from models.model_module import *

from utils import *

# Convolutional Neural Network (base class)
class NetworkCNN(nn.Module):
    def __init__(self,
                 patch_shape,
                 flow_shape,
                 config,
                 dtype = torch.float32):
        super(NetworkCNN, self).__init__()

        # Define a name for this network
        self.name = config["network_name"]

        # Define the data type
        self.dtype = dtype

        # Define patch shape
        self.patch_nb, self.patch_nc, self.patch_nx, self.patch_ny, self.patch_nz = patch_shape

        # Define flow shape
        (self.flow_nb, self.flow_nc, self.flow_nx, self.flow_ny, self.flow_nz) = flow_shape

        # Prepare the convolutional layer module
        self.conv_block = self.build_conv_layers(config = config)

        # Prepare the fully connected module
        self.fc_block = self.build_fc_layers(config = config)

        # Get the maximum projection
        self.proj_max_flow = config["proj_max_flow"]

        # Store the config
        self.config = config

        return
    
    def build_conv_layers(self, config):
        # Prepare the convolutional layer module
        conv_block = nn.Sequential()
        
        # For each convolutional layer,
        for idx, conv_layer in enumerate(config["conv_layers"]):
            # Add convolutional layer
            conv_block.add_module(f'conv_{idx}',
                                  nn.Conv3d(in_channels = conv_layer["in_channels"],
                                            out_channels = conv_layer["out_channels"],
                                            kernel_size = conv_layer["kernel_size"],
                                            stride = conv_layer["stride"],
                                            padding = conv_layer["padding"],
                                            groups = conv_layer["groups"],
                                            dtype = self.dtype))

            # Add Normalization layer
            conv_block.add_module(f'conv_norm_{idx}',
                                  ModuleNorm(num_channels = conv_layer["out_channels"],
                                             norm_type = config["conv_layers_norm"],
                                             dtype = self.dtype))
            
            # Add Activation layer
            conv_block.add_module(f'conv_acti_{idx}', 
                                  ModuleActi(acti_type = config["conv_layers_acti"]))
            
            # Add pooling layer
            conv_block.add_module(f'conv_pool_{idx}',
                                  ModulePool(kernel_size = conv_layer["pool_size"],
                                             pool_type = config["conv_layers_pool"],
                                             dtype = self.dtype))
            
            # Add dropout layer
            conv_block.add_module(f'conv_dropout_{idx}',
                                  ModuleDropout(p = config["conv_layers_dropout_p"],
                                                dropout_type = config["conv_layers_dropout"]))
            

        return conv_block
    
    def build_fc_layers(self, config):
        # Prepare the fully connected module
        fc_block = nn.Sequential()

        # Input size of FC layers
        self.fc_input_size = self.calculate_fc_input_size( \
            input_size = (self.patch_nc * 2, 
                          self.patch_nx, 
                          self.patch_ny, 
                          self.patch_nz), 
            config = config)
        
        print(f'fc_input_size = {self.fc_input_size}')
        
        # Get input size of each layer
        input_size = self.fc_input_size

        ## TODO: automatically calculate the output size of FC layers

        # For each fully connected layer
        for idx, fc_size in enumerate(config["fc_sizes"]):
            # Add fully connected layer
            fc_block.add_module(f'fc_{idx}',
                                nn.Linear(in_features = input_size,
                                          out_features = fc_size,
                                          dtype = self.dtype))
            
            # Add Normalization layer
            fc_block.add_module(f'fc_norm_{idx}',
                                ModuleNorm(num_channels = fc_size,
                                           norm_type = config["fc_layers_norm"],
                                           dtype = self.dtype))
            
            # Add Activation layer
            fc_block.add_module(f'fc_acti_{idx}', 
                                ModuleActi(acti_type = config["fc_layers_acti"]))

            # Add dropout layer
            fc_block.add_module(f'fc_dropout_{idx}',
                                ModuleDropout(p = config["fc_layers_dropout_p"],
                                              dropout_type = config["fc_layers_dropout"]))
            
            # The next input size is the current output size
            input_size = fc_size

        return fc_block
    
    def calculate_fc_input_size(self, input_size, config):
        # Dummy input to get the size after convolutional layers
        dummy_input = torch.randn((1, *input_size))
        with torch.no_grad():
            conv_output = self.build_conv_layers(config)(dummy_input)
        return conv_output.view(1, -1).size(1)
    
    # Get network summary as a string
    def print_network_summary(self, device, patch_shape) -> None:
        _, nc, nx, ny, nz = patch_shape
            
        input_vol0 = torch.randn(nz, nc, nx, ny, nz).to(device)
        input_vol1 = torch.randn(nz, nc, nx, ny, nz).to(device)

        with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
            self.forward(input_vol0, input_vol1)

        str_summary_prof = prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=100)
        input_size = [(nc, nx, ny, nz), 
                      (nc, nx, ny, nz)]
        
        print(str_summary_prof)
        
        # str_summary_net, _ = summary_string(self, input_size = input_size)
        # str_summary = "{}\n\n{}".format(str_summary_prof, str_summary_net)
        summary(self, 
                input_size = input_size, 
                device = device)

        return
    
    def initialize_weights(self, mode='zero'):
        # Initialize convolutional layers
        for m in self.conv_block.modules():
            if isinstance(m, nn.Conv3d):
                # Initialize all weights
                if  mode.lower() == 'zero':
                    nn.init.constant_(m.weight, 0.0)
                if mode.lower() == 'kaiming':
                    nn.init.kaiming_normal_(m.weight)

                # Initialize all of the bias to 0
                nn.init.constant_(m.bias, 0.0)

        # Initialize fully connected layers
        for m in self.fc_block.modules():
            if isinstance(m, nn.Linear):
                # Initialize all weights
                if  mode.lower() == 'zero':
                    nn.init.constant_(m.weight, 0.0)
                if mode.lower() == 'kaiming':
                    nn.init.kaiming_normal_(m.weight)

                # Initialize all of the bias to 0
                nn.init.constant_(m.bias, 0.0)

    # Projection
    def projection(self, out):
        # project to certain range
        return out * self.proj_max_flow
    
    # Clip gradient norm
    def clip_gradient_norm(self):
        if 'gradient_norm_clip' in self.config.keys():
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                           max_norm = self.config["gradient_norm_clip"])
        
        return
    
    # normalize the range across both tensors
    # tensor structure should be: (Batch, Channel, Height, Width, Depth)
    def normalization(self, tensor0, tensor1):
        b0, c0, h0, w0, d0 = tensor0.shape
        b1, c1, h1, w1, d1 = tensor1.shape

        assert b0 == b1, f'batch size is not matching between {tensor0.shape} and {tensor1.shape}'
        assert c0 == c1, f'channel size is not matching between {tensor0.shape} and {tensor1.shape}'
        assert (h0 == h1) & (w0 == w1) & (d0 == d1), f'volume size is not matching between {tensor0.shape} and {tensor1.shape}'

        # Stack tensors to the last dimension
        tensor = torch.stack((tensor0, tensor1), dim = 5)

        # Compute the sd and mean at (height, width, depth) and the last dimension
        tensor_min = torch.amin(tensor, dim=(2, 3, 4, 5), keepdim = True)
        tensor_max = torch.amax(tensor, dim=(2, 3, 4, 5), keepdim = True)

        # Squeeze out the last dimension
        # Repeat in the height, width and depth dimensions
        tensor_min = tensor_min.squeeze(-1).repeat([1, 1, h0, w0, d0])
        tensor_max = tensor_max.squeeze(-1).repeat([1, 1, h0, w0, d0])

        # Normalize both tensors by the min and max to [0, 1]
        tensor0_norm = (tensor0 - tensor_min)
        tensor1_norm = (tensor1 - tensor_min)
        mask = (tensor_max - tensor_min > 0.0)
        tensor0_norm[mask] = tensor0_norm[mask] / (tensor_max - tensor_min)[mask]
        tensor1_norm[mask] = tensor1_norm[mask] / (tensor_max - tensor_min)[mask]

        tensor0_norm = tensor0_norm.contiguous()
        tensor1_norm = tensor1_norm.contiguous()

        return tensor0_norm, tensor1_norm

    def forward(self, vol0, vol1):
        # Normalize for vol0 and vol1
        vol0, vol1 = self.normalization(vol0, vol1)
        
        # Merge vol0 and vol1 into a single volume tensr with 2 channels
        vol = torch.concat((vol0, vol1), dim = 1)

        # Apply Sequential Module
        feature = self.conv_block(vol)

        # Prepare for fully connected layers
        feature = feature.view(-1, self.fc_input_size)

        # Apply fully connected layers
        feature = self.fc_block(feature)

        # Reshape the output
        out = feature.view(-1, self.flow_nc, self.flow_nx, self.flow_ny, self.flow_nz)

        return self.projection(out)

class ModelCNN(Model):
    def __init__(self,
                 patch_shape,
                 flow_shape,
                 config,
                 dtype = torch.float32):
        super().__init__()

        self.network = NetworkCNN(patch_shape, flow_shape, config, dtype)
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

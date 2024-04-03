import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib as mpl

from torch.utils.data import random_split, DataLoader, RandomSampler

from utils import *
from datasets import *
from models import *

'''
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
Functions
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
'''

'''
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
Program
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
'''
def setup_environment(args):
    """
    Setup the training environment based on the provided arguments
    """
    # Setup Logger
    logger = Logger(jobid=args.jobid, name='debug_train' if args.debug else 'train', verbose=args.debug)

    logger.build_output(output_folder_path = args.output_folder)
    logger.setup()  # Redirect stdout to logger

    # Load JobID
    if args.jobid:
        print(f'JobID = {args.jobid}')

    # Load version of packages
    print_versions()

    # Load configuration if provided
    if args.config:
        # Load config
        config = YAMLHandler.read_yaml(args.config)
        print(f'load configuration from {args.config}')
        logger.save_config(config_path = args.config, 
                           verbose = args.debug)
    else:
        raise ValueError(f'Error occurs when loading config from {args.config}')

    # Setup device
    device_manager = DeviceManager(verbose = False)  # Automatically select the best available device
    device = device_manager.get_device(use_cpu = args.cpu)
    print(f"Using device: {device}")

    ptdtype = torch.float32 # PyTorch data type
    npdtype = np.float32 # Numpy data type

    # Setup seed
    seed_manager = SeedManager(seed = config["seed"])
    print(f"Seed: {seed_manager.get_current_seed()}")

    # Setup matplotlib
    plot_manager = PlotManager(figsize=(16, 12))

    mpl.use('Agg')
    print(f'matplotlib backend is: {mpl.get_backend()}')

    return logger, device_manager, device, seed_manager, plot_manager, config, ptdtype, npdtype

def data_preparation(args, config):
    """
    Prepare the data
    """
    # get the patch_shape from config
    patch_shape = config.get("patch_shape", config.get("patch_xyz"))

    # Build dataset instance
    dataset = DVCPatchDataset(dataset_path = args.dataset_path,
                              patch_shape = patch_shape)
    dataset.fetch()
    
    # Prepare patch shape for the volumes
    patch_shape = (config["minibatch_size"], dataset.nc, dataset.nx, dataset.ny, dataset.nz)
    print(f'dataset length = {len(dataset)}')
    print(f'patch_shape = {patch_shape}')

    # Prepare flow shape for the predicted flow
    if config["flow_shape"] is None:
        flow_shape = [config["minibatch_size"], 
                      3, 
                      1, 
                      1, 
                      1]
    else:
        flow_shape = [config["minibatch_size"], 
                      3, 
                      config["flow_shape"][0],
                      config["flow_shape"][1],
                      config["flow_shape"][2]]
        
    flow_margin_predict = [(0, 0, 0), 
                           (0, 0, 0)]
    
    flow_margin_target = [
        ((patch_shape[2] - flow_shape[2] + 1) // 2, 
         (patch_shape[3] - flow_shape[3] + 1) // 2, 
         (patch_shape[4] - flow_shape[4] + 1) // 2), 
        ((patch_shape[2] - flow_shape[2]) // 2, 
         (patch_shape[3] - flow_shape[3]) // 2, 
         (patch_shape[4] - flow_shape[4]) // 2)
         ]
    print(f'flow_shape = {flow_shape}')
    print(f'flow_margin_predict = {flow_margin_predict}')
    print(f'flow_margin_target = {flow_margin_target}')

    # Split to train and valid datasets
    size_train = int(config["dataset_size_train"] * len(dataset))
    size_valid = len(dataset) - size_train
    dataset_train, dataset_valid = random_split(dataset, [size_train, size_valid])
    print(f'dataset_train length = {len(dataset_train)}')
    print(f'dataset_valid length = {len(dataset_valid)}')

    # Get number of epochs
    epochs = int(config["epochs"])
    print(f'number of epochs = {epochs}')

    # Define dataloaders
    sampler_train = RandomSampler(dataset_train, 
                                  num_samples = int(config["minibatch_size"]) * int(config["minibatch_num_train"]),
                                  replacement = True)

    dataloader_train = DataLoader(dataset_train,
                                  batch_size = config["minibatch_size"],
                                  num_workers = config["num_workers_train"],
                                  pin_memory = True,
                                  sampler = sampler_train)
    print(f'dataloader_train length = {len(dataloader_train)}')
    print(f'dataloader_train minibatch length = {int(config["minibatch_num_train"])}')
    print(f'num_workers_train = {config["num_workers_train"]}')

    sampler_valid = RandomSampler(dataset_valid, 
                                  num_samples = int(config["minibatch_size"]) * int(config["minibatch_num_valid"]),
                                  replacement = True)
    dataloader_valid = DataLoader(dataset_valid,
                                  batch_size = config["minibatch_size"],
                                  num_workers = config["num_workers_valid"],
                                  pin_memory = True,
                                  sampler = sampler_valid)
    print(f'dataloader_valid length = {len(dataloader_valid)}')
    print(f'dataloader_train minibatch length = {int(config["minibatch_num_valid"])}')
    print(f'num_workers_valid = {config["num_workers_valid"]}')
    
    return \
        dataset, dataset_train, dataset_valid, \
            patch_shape, \
            flow_shape, flow_margin_predict, flow_margin_target, \
            epochs, \
            dataloader_train, dataloader_valid

def model_preparation(args, config, device, patch_shape, flow_shape, flow_margin_predict, flow_margin_target, ptdtype = torch.float32) -> tuple[Model, optim.Optimizer, optim.lr_scheduler.LRScheduler, nn.Module]:
    """
    Prepare the model
    """
    # Define network
    model = ModelFactory.build_instance(patch_shape = patch_shape, 
                                        flow_shape = flow_shape, 
                                        config = config, 
                                        ptdtype = ptdtype)
    
    # Load checkpoint
    if args.checkpoint is None:
        # Initialize network
        model.initialize_weights(mode = 'kaiming')
    else:
        # Load checkpoint
        checkpoint_controller = CheckpointController(checkpoint_dir = args.checkpoint)

        epoch_idx, patch_shape, flow_shape, model, optimizer, scheduler, loss_list_train, loss_list_valid =\
            checkpoint_controller.load_last_checkpoint(network = model, 
                                                       optimizer = optimizer, 
                                                       scheduler = scheduler)
    
    model = model.to(device)
    model.print_network_summary(device = device, patch_shape = patch_shape)

    if model.name is not None:
        print(f'model name is {model.name}')

    optimizer = OptimizerFactory.build_instance(config, model)

    scheduler = SchedulerFactory.build_instance(config, optimizer)

    criterion = CriterionFactory.build_instance(config = config,
                                                device = device,
                                                ptdtype = ptdtype,
                                                flow_margin_predict = flow_margin_predict,
                                                flow_margin_target = flow_margin_target)
    
    return model, optimizer, scheduler, criterion

def train(args) -> None:
    """
    Main training function.
    """
    logger, device_manager, device, seed_manager, plot_manager, config, ptdtype, npdtype = setup_environment(args)

    dataset, dataset_train, dataset_valid, \
        patch_shape, \
        flow_shape, flow_margin_predict, flow_margin_target, \
        epochs, \
        dataloader_train, dataloader_valid  = data_preparation(args, config)
    
    model, optimizer, scheduler, criterion = model_preparation(args, config, device, patch_shape, flow_shape, flow_margin_predict, flow_margin_target, ptdtype)
    
    model.train(args = args, 
                config = config, 
                epochs = epochs,
                logger = logger,
                plot_manager = plot_manager,
                dataloader_train = dataloader_train, 
                dataloader_valid = dataloader_valid, 
                patch_shape = patch_shape,
                flow_shape = flow_shape,
                optimizer = optimizer, 
                scheduler = scheduler, 
                criterion = criterion,
                device = device, 
                ptdtype = ptdtype)

    logger.teardown()  # Restore stdout

'''
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
Main function
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
'''
if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser(description='PyTorch Neural Network Training')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset')
    parser.add_argument("--config", type=str, help="Path to the configuration file (YAML format)")
    parser.add_argument('--jobid', type=str, default=None, help='Job identifier')
    parser.add_argument('--output_folder', type=str, default='./output', help='Output folder path')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint for resuming training')
    parser.add_argument('--file_mode', type=int, default=1, help='File loading mode for dataset. Default: 1 (JSON)')
    
    parser.add_argument('--cpu', action='store_true', help='Force using CPU for training')
    parser.add_argument('--debug', action='store_true', help="Enable debug logging mode")
    parser.add_argument('--verbose_batch', action='store_true', help="Enable verbose mode for batch processing")
    
    args = parser.parse_args()

    # Ensure the output directory exists
    FileHandler.mkdir(args.output_folder)

    # Start training
    train(args)

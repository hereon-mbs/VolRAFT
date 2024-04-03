# Prepare data
import argparse
import os
import numpy as np
import torch
import matplotlib as mpl

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
def normalize_data_by_mask(data_np, mask_np):
    # Applying the mask to select only valid elements
    valid_data = data_np[mask_np]

    # Normalizing only valid elements to range [0, 1]
    data_min = valid_data.min()
    data_max = valid_data.max()

    # return the normalized range
    return (data_np - data_min) / (data_max - data_min)

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
    logger = Logger(jobid=args.jobid, name='debug_data' if args.debug else 'data', verbose=args.debug)

    logger.build_output(output_folder_path = args.output_folder, 
                        make_checkpoint = False)
    logger.setup()  # Redirect stdout to logger

    # Load JobID
    if args.jobid:
        print(f'JobID = {args.jobid}')

    # Load version of packages
    print_versions()

    # Setup device
    device_manager = DeviceManager(verbose = False)  # Automatically select the best available device
    device = device_manager.get_device(use_cpu = args.cpu)
    print(f"Using device: {device}")

    ptdtype = torch.float32 # PyTorch data type
    npdtype = np.float32 # Numpy data type

    # Setup seed
    seed_manager = SeedManager(seed = args.seed)
    print(f"Seed: {seed_manager.get_current_seed()}")

    # Setup matplotlib
    plot_manager = PlotManager(figsize=(16, 12))

    mpl.use('Agg')
    print(f'matplotlib backend is: {mpl.get_backend()}')

    return logger, device_manager, device, seed_manager, plot_manager, ptdtype, npdtype

def prepare_base_volumes(args):
    # Load vol0
    vol0 = DVCVolume()
    vol0.load_data(folder_path = args.vol0_source_path,
                   tiff_subfolder = 'registration')
    vol0.to_dhw()

    # Load mask
    mask = DVCMask()
    mask.load_data_labels(folder_path= args.mask_source_path,
                          labels = args.mask_label,
                          tiff_subfolder = 'watershed',
                          closing_percent = 0.0,
                          npdtype = np.bool_)
    mask.to_dhw()

    # Load mask_fill
    mask_fill = DVCMask()
    mask_fill.load_data_labels(folder_path= args.mask_source_path,
                               labels = args.mask_label,
                               tiff_subfolder = 'watershed',
                               closing_percent = args.mask_closing_percent,
                               npdtype = np.bool_)
    mask_fill.to_dhw()

    return vol0, mask, mask_fill

def prepare_volumes(args, device, vol0, mask, mask_fill):
    def volume_warping(vol0, device, ptdtype=torch.float32):
        vol0_tensor = torch.from_numpy(vol0.data_np).to(device, 
                                                        dtype=ptdtype, 
                                                        non_blocking=False).requires_grad_(False)
        flow_tensor = torch.from_numpy(flow.data_np).to(device, 
                                                        dtype=ptdtype, 
                                                        non_blocking=False).requires_grad_(False)
        
        vol1_tensor = VolumeWarping.warp(vol0_tensor, flow_tensor, device = device)

        # Clear tensor immediately
        del vol0_tensor
        vol0_tensor = None
        del flow_tensor
        flow_tensor = None
        DeviceManager.garbage()

        vol1 = DVCVolume(volume_data= vol1_tensor.squeeze().cpu().numpy(),
                         indexing = vol0.indexing,
                         npdtype = vol0.npdtype)
        
        # Clear tensor immediately
        del vol1_tensor
        vol1_tensor = None
        DeviceManager.garbage()

        return vol1
    
    def add_noise(vol, mask) -> np.ndarray:
        # AWGN
        # Estimated by the square of background magnitude
        awgn_noise_level = (0.000150) ** 2

        # Shot noise
        # 0.008 percent
        shot_noise_level = 0.01 * 0.008

        vol_awgn, _ = DVCDataAugmentation.add_awgn(vol, noise_power = awgn_noise_level)

        vol_noisy, _ = DVCDataAugmentation.add_shot_noise(vol_awgn, mask = mask, intensity = shot_noise_level)
        
        return vol_noisy.astype(vol.dtype)

    # Generate flow
    flow = DVCFlowGenerator.generate_synthetic(shape = vol0.shape,
                                               indexing = vol0.indexing,
                                               npdtype = vol0.npdtype,
                                               flow_config = args.flow_config,
                                               mask = mask_fill)

    # Warp vol0 to vol1 by the flow
    vol1 = volume_warping(vol0, device = device)
    
    ## Add Noise to vol0 and vol1
    if "add_noise" in args.flow_config:
        print(f'start add noise')
        vol0 = DVCVolume(add_noise(vol0.data, mask = mask.data),
                         indexing = vol0.indexing,
                         npdtype = vol0.npdtype)

        vol1 = DVCVolume(add_noise(vol1.data_np, mask = mask.data_np),
                         indexing = vol1.indexing,
                         npdtype = vol1.npdtype)
        print(f'finish add noise')
    
    DeviceManager.garbage()

    return vol1, flow

def save_base_volumes_npy(args, 
                          vol0_target_path,
                          mask_target_path,
                          mask_fill_target_path,
                          vol0 : DVCVolume, 
                          mask : DVCMask, 
                          mask_fill : DVCMask):
    """
    Save vol0, mask, mask_fill to the base folder
    """
    # Make folders
    FileHandler.mkdir(vol0_target_path)
    FileHandler.mkdir(mask_target_path)
    FileHandler.mkdir(mask_fill_target_path)

    # Get dimension
    shape = mask.shape
    dimension = f'{shape[0]}x{shape[1]}x{shape[2]}'

    # Get all paths
    vol0_path_base = FileHandler.join(vol0_target_path, f'vol0_{dimension}.npy')
    mask_path_base = FileHandler.join(mask_target_path, f'mask_{dimension}.npy')
    mask_fill_path_base = FileHandler.join(mask_fill_target_path, f'mask_fill_{dimension}.npy')

    # Save to files
    np.save(vol0_path_base, DVCVolume.add_channel_dims(vol0.data_np))
    np.save(mask_path_base, DVCVolume.add_channel_dims(mask.data_np).astype(bool))
    np.save(mask_fill_path_base, DVCVolume.add_channel_dims(mask_fill.data_np).astype(bool))

    # Return paths
    return vol0_path_base, mask_path_base, mask_fill_path_base

def save_volumes_npy(args, vol1 : DVCVolume, flow : DVCFlow, volume_base_path) -> None:
    """
    Save volumes (mask, vol0, vol1, flow) to npy files
    """
    # Prepare volume folders
    volume_dataset_path = FileHandler.join(args.volume_path, args.name)
    
    # Make folders
    FileHandler.mkdir(FileHandler.join(volume_dataset_path, "volume1"))
    FileHandler.mkdir(FileHandler.join(volume_dataset_path, "ground_truth", "flow"))

    # Make links
    FileHandler.ln(FileHandler.join(volume_base_path, 'volume0'), 
                   FileHandler.join(volume_dataset_path, "volume0"))
    FileHandler.ln(FileHandler.join(volume_base_path, 'mask'), 
                   FileHandler.join(volume_dataset_path, "mask"))
    FileHandler.ln(FileHandler.join(volume_base_path, 'mask_fill'), 
                   FileHandler.join(volume_dataset_path, "mask_fill"))
    
    # Get dimensions
    shape = vol1.shape
    dimension = f'{shape[0]}x{shape[1]}x{shape[2]}'

    # Get all paths
    mask_path = FileHandler.join(volume_dataset_path, "mask_fill")
    vol0_path = FileHandler.join(volume_dataset_path, "volume0")
    vol1_path = FileHandler.join(volume_dataset_path, "volume1")
    flow_path = FileHandler.join(volume_dataset_path, "ground_truth", "flow")

    # Save to files
    np.save(FileHandler.join(vol1_path, f'vol1_{dimension}.npy'), 
            DVCVolume.add_channel_dims(vol1.data_np))
    np.save(FileHandler.join(flow_path, f'flow_{dimension}.npy'), 
            flow.data_np)

    # Return the paths
    return mask_path, vol0_path, vol1_path, flow_path

def save_volume_to_patches(data_np, mask, args, verbose = True):
    """
    Save volumes into patches
    """
    # Check if data is in correction dimension
    assert data_np.ndim == 4, 'data_np should have 4 dimensions: (C, D, H, W)'

    # Prepare the shape
    shape = data_np.shape

    # Prepare the folder
    FileHandler.mkdir(args.target_path)

    # Prepare a list to store all grids
    grids_all = list()

    # Define grid
    for axis in [1, 2, 3]:
        grid_starts = np.arange(start = 0, 
                                stop = shape[axis], 
                                step = args.patch_stride[axis],
                                dtype = int)

        grids = list()
        for idx in range(len(grid_starts)):
            grid_start = grid_starts[idx]
            grid_end = grid_start + args.patch_size[axis]
            if grid_end >= shape[axis]:
                grid_end = shape[axis]
                grid_start = grid_end - args.patch_size[axis]
                grids.append([grid_start, grid_end])
                break
            else:
                grids.append([grid_start, grid_end])

        grids_all.append(grids)

    # Prepare grids x, y, z
    grids_x = grids_all[0]
    grids_y = grids_all[1]
    grids_z = grids_all[2]

    # Prepare all patches
    count = 0
    for _, grid_x in enumerate(grids_x):
        for _, grid_y in enumerate(grids_y):
            for _, grid_z in enumerate(grids_z):
                # Extract the patch by the grids
                patch_mask = mask.data_np[grid_x[0]:grid_x[1], grid_y[0]:grid_y[1], grid_z[0]:grid_z[1]]

                # if patch of mask contains any valid (True) pixels
                if patch_mask.any():
                    # Extract volume0
                    patch = data_np[:, grid_x[0]:grid_x[1], grid_y[0]:grid_y[1], grid_z[0]:grid_z[1]].astype(args.npdtype)

                    # Get the full path of file
                    dimension = f"{args.patch_size[1]}x{args.patch_size[2]}x{args.patch_size[3]}"
                    filepath = FileHandler.join(args.target_path, f"{args.prefix}_{count:0>{args.num_zero}}_{dimension}.npy")

                    # Get the folder path
                    folder = os.path.dirname(os.path.abspath(filepath))

                    # Prepare the folder if not existed
                    if os.path.isdir(folder) is False:
                        os.makedirs(folder, mode = 0o711, exist_ok = True)

                    # Save to file
                    np.save(filepath, patch)
                    
                    if verbose:
                        print(f'saved to {filepath}')

                    count += 1

    return

def save_base_patches_npy(args, 
                          patch_base_path, 
                          mask_fill : DVCMask, 
                          vol0 : DVCVolume):
    """
    Save vol0, mask_fill to the base folder
    """
    # Normalize both vol0 and vol1
    vol0_norm = normalize_data_by_mask(vol0.data, mask_fill.data)

    # Make folders
    FileHandler.mkdir(patch_base_path)

    # Prepare arguments
    args_patch = argparse.Namespace()
    args_patch.patch_size = args.patch_size
    args_patch.patch_stride = args.patch_stride
    args_patch.num_zero = args.num_zero

    # Get all paths
    # vol0_path_base = FileHandler.join(vol0_target_path, f'vol0_{dimension}.npy')
    # mask_fill_path_base = FileHandler.join(mask_fill_target_path, f'mask_fill_{dimension}.npy')

    # Save mask
    args_patch.target_path = FileHandler.join(patch_base_path, "mask_fill")
    args_patch.prefix = "pmask"
    args_patch.npdtype = mask_fill.npdtype
    save_volume_to_patches(data_np = DVCVolume.add_channel_dims(mask_fill.data_np),
                           mask = mask_fill,
                           args = args_patch,
                           verbose = False)
    
    # Save vol0
    args_patch.target_path = FileHandler.join(patch_base_path, "norm_volume0")
    args_patch.prefix = "pvol0"
    args_patch.npdtype = vol0.npdtype
    save_volume_to_patches(data_np = DVCVolume.add_channel_dims(vol0_norm),
                           mask = mask_fill,
                           args = args_patch,
                           verbose = False)


    # Return paths
    return

def save_patches_npy(args, patch_base_path, mask_fill : DVCMask, vol1 : DVCVolume, flow : DVCFlow):
    """
    Save all patches to npy files
    """
    # Normalize vol1
    vol1_norm = normalize_data_by_mask(vol1.data, mask_fill.data)

    # Prepare volume folders
    patch_dataset_path = FileHandler.join(args.patch_path, args.name)

    # Make folders
    FileHandler.mkdir(patch_dataset_path)
    FileHandler.mkdir(FileHandler.join(patch_dataset_path, "norm_volume1"))
    FileHandler.mkdir(FileHandler.join(patch_dataset_path, "ground_truth", "flow"))

    # Prepare arguments
    args_patch = argparse.Namespace()
    args_patch.patch_size = args.patch_size
    args_patch.patch_stride = args.patch_stride
    args_patch.num_zero = args.num_zero

    # Save mask using symbolic link
    mask_path_patch = args_patch.target_path = FileHandler.join(patch_dataset_path, "mask_fill")
    FileHandler.ln(FileHandler.join(patch_base_path, "mask_fill"), mask_path_patch)
    
    # Save vol0 using symbolic link
    vol0_path_patch = args_patch.target_path = FileHandler.join(patch_dataset_path, "norm_volume0")
    FileHandler.ln(FileHandler.join(patch_base_path, "norm_volume0"), vol0_path_patch)
    
    # Save vol1
    vol1_path_patch = args_patch.target_path = FileHandler.join(patch_dataset_path, "norm_volume1")
    args_patch.prefix = "pvol1"
    args_patch.npdtype = vol1.npdtype
    save_volume_to_patches(data_np = DVCVolume.add_channel_dims(vol1_norm),
                           mask = mask_fill,
                           args = args_patch,
                           verbose = False)
    
    # Save flow
    flow_path_patch = args_patch.target_path = FileHandler.join(patch_dataset_path, "ground_truth", "flow")
    args_patch.prefix = "pflow"
    args_patch.npdtype = flow.npdtype
    save_volume_to_patches(data_np = flow.data_np,
                           mask = mask_fill,
                           args = args_patch,
                           verbose = False)

    return mask_path_patch, vol0_path_patch, vol1_path_patch, flow_path_patch

def prepare_data_new(args) -> None:
    '''
    Main prepare data function
    '''
    logger, device_manager, device, seed_manager, plot_manager, ptdtype, npdtype = setup_environment(args)

    # Load the content of YAML files
    content_measure = YAMLHandler.read_yaml(args.measurement_path)
    content_flow = YAMLHandler.read_yaml(args.flow_path)

    # Prepare the overall content of YAML files for saving
    content_datasets = {}
    content_datasets["datasets_train"] = []

    # for each measurement
    for idx_m, measurement in enumerate(content_measure["measurements"]):
        # Load base volumes
        args_base = args
        args_base.volume_path = args.dest_volume_folder
        args_base.patch_path = args.dest_patch_folder
        args_base.mask_source_path = FileHandler.join(measurement["path"])
        args_base.vol0_source_path = FileHandler.join(measurement["path"])
        args_base.mask_label = content_measure["mask_label"]
        args_base.mask_closing_percent = content_measure["mask_closing_percent"]
        args_base.num_zero = 6
        
        # vol0_source_path = FileHandler.join(measurement["path"])
        vol0, mask, mask_fill = prepare_base_volumes(args_base)

        # Prepare folders
        volume_base_path = FileHandler.join(args_base.volume_path, f'{measurement["name"]}')
        patch_base_path = FileHandler.join(args_base.patch_path, f'{measurement["name"]}')

        # Save base volumes
        save_base_volumes_npy(args_base,
                              FileHandler.join(volume_base_path, 'volume0'),
                              FileHandler.join(volume_base_path, 'mask'),
                              FileHandler.join(volume_base_path, 'mask_fill'),
                              vol0, mask, mask_fill)
        
        # Save base patches
        save_base_patches_npy(args_base,
                              patch_base_path,
                              mask_fill,
                              vol0)

        # for each flow
        for idx_f, flow in enumerate(content_flow["flows"]):
            # Load arguments from the contents
            args_dataset = args

            args_dataset.name = f'{measurement["name"]}_{flow["name"]}'
            args_dataset.volume_path = args.dest_volume_folder
            args_dataset.patch_path = args.dest_patch_folder
            args_dataset.mask_label = content_measure["mask_label"]
            args_dataset.mask_closing_percent = content_measure["mask_closing_percent"]
            args_dataset.flow_config = dict(flow)
            args_dataset.num_zero = args_base.num_zero

            print(f'-----------------------------------------------------------')
            print(f'{idx_m:03d},{idx_f:03d}: {args_dataset.name} start')

            vol1, flow = prepare_volumes(args_dataset, device, vol0, mask, mask_fill)

            # Save volumes
            mask_path, vol0_path, vol1_path, flow_path = save_volumes_npy(args_dataset, vol1, flow, volume_base_path)
            DeviceManager.garbage()

            # Save patches
            mask_path_patch, vol0_path_patch, vol1_path_patch, flow_path_patch = save_patches_npy(args_dataset, 
                                                                                                  patch_base_path,
                                                                                                  mask_fill, 
                                                                                                  vol1, 
                                                                                                  flow)
            DeviceManager.garbage()
            
            dataset = {
                "name": args_dataset.name,
                "enable": True,
                "mask_path": mask_path_patch,
                "vol0_path": vol0_path_patch,
                "vol1_path": vol1_path_patch,
                "flow_path": flow_path_patch
                }

            content_datasets["datasets_train"] += [dataset]
            print(f'{idx_m:03d},{idx_f:03d}: {args_dataset.name} done')
            YAMLHandler.write_yaml(args.target_path, content_datasets)

    return

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
    parser = argparse.ArgumentParser(description='PyTorch Prepare data')
    parser.add_argument('measurement_path', type=str, help='File path to the YAML file of measurement datasets path')
    parser.add_argument('flow_path', type=str, help='File path to the YAML file of flow field path')
    parser.add_argument('target_path', type=str, help='File path to the YAML file of the target')

    parser.add_argument('dest_volume_folder', type=str, help='Destination folder path of the volume')
    parser.add_argument('dest_patch_folder', type=str, help='Destination folder path of the patch')

    parser.add_argument('--patch_size', type=int, nargs='+', default=[1, 60, 80, 80], help='The target patch size in (Channels, Depth, Height, Width) format')
    parser.add_argument('--patch_stride', type=int, nargs='+', default=[1, 30, 40, 40], help='The target patch stride in (Channels, Depth, Height, Width) format')
    parser.add_argument('--jobid', type=str, default=None, help='String that represent the job')
    parser.add_argument('--output_folder', type=str, default='./output', help='Folder path of the output')
    parser.add_argument('--seed', type=int, default=0, help='Integer that represents the seed')
    
    parser.add_argument('--cpu', action='store_true', help='Enable CPU for training')
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    args = parser.parse_args()

    # Training the model
    # prepare_data(args = None)
    # prepare_all_data(args)
    prepare_data_new(args)
   
    # end of main function
    
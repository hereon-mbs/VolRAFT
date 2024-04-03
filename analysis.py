# Analysis
import argparse

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

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
def epe(predict, target, axis=1, mask=None):
    """
    Calculate the End-Point Error (EPE) for optical flow with a specified axis and an optional mask.

    :param predict: Predicted optical flow (Tensor or NumPy array)
    :param target: Ground truth optical flow (Tensor or NumPy array)
    :param axis: Axis along which the flow field is represented (default is 1)
    :param mask: Optional binary mask (Tensor or NumPy array) to indicate valid positions
    :return: EPE (float), EPE_MAP (NumPy Array)
    """

    if isinstance(predict, torch.Tensor) and isinstance(target, torch.Tensor):
        # PyTorch computation
        difference = target - predict
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise TypeError("For PyTorch tensors, mask must also be a PyTorch tensor")
            # difference = difference * mask.bool()
        
        distance = torch.sqrt(torch.sum(difference ** 2, dim=axis))
        epe_value = torch.mean(distance[mask.bool() == 1]).item() if mask is not None else torch.mean(distance).item()
        epe_map = distance.numpy()

    elif isinstance(predict, np.ndarray) and isinstance(target, np.ndarray):
        # NumPy computation
        difference = target - predict
        if mask is not None:
            if not isinstance(mask, np.ndarray):
                raise TypeError("For NumPy arrays, mask must also be a NumPy array")
            # difference = difference * mask.astype(bool)
        
        distance = np.sqrt(np.sum(difference ** 2, axis=axis))
        epe_value = np.mean(distance[mask.astype(bool) == 1]) if mask is not None else np.mean(distance)
        epe_map = distance
        
    else:
        raise TypeError("Input types must be both PyTorch Tensors or both NumPy arrays")

    return epe_value, epe_map

def operation(arr):
    return np.transpose(arr)

def mse(gt, measure, mask):
    # Use Mean-Square-Error
    # return np.mean((gt * mask - measure * mask) ** 2)
    if np.sum(mask) > 0.0:
        return np.sum((gt * mask - measure * mask) ** 2) / np.sum(mask)
    else:
        return -1.0

def mae(gt, measure, mask):
    # Use Mean-Absolute-Error
    # return np.mean(np.abs(gt * mask - measure * mask))
    if np.sum(mask) > 0.0:
        return np.sum(np.abs(gt * mask - measure * mask)) / np.sum(mask)
    else:
        return -1.0
    
def rmse(gt, measure, mask):
    # Use Mean-Square-Error
    # return np.mean((gt * mask - measure * mask) ** 2)
    if np.sum(mask) > 0.0:
        return np.sqrt(np.sum((gt * mask - measure * mask) ** 2) / np.sum(mask))
    else:
        return -1.0

def find_percentiles(volumes, mask, percentiles=(10, 90)):
    """
    Find specified percentiles across a list of volumes based on a mask.

    :param volumes: List of 2D NumPy arrays
    :param mask: 2D boolean NumPy array
    :param percentiles: A tuple of two numbers indicating the percentiles to calculate
    :return: A tuple containing the specified percentiles
    """

    # Validate that all volumes have the same shape as the mask
    for vol in volumes:
        if vol.shape != mask.shape:
            raise ValueError("All volumes must have the same shape as the mask")

    # Flatten and filter each volume using the mask, then concatenate
    combined_vols = np.concatenate([vol.flatten()[mask.flatten()] for vol in volumes])

    # Calculate the specified percentiles
    p1 = np.percentile(combined_vols, percentiles[0])
    p2 = np.percentile(combined_vols, percentiles[1])

    return p1, p2

# slice
def slice_at(arr, index, axis):
    return operation(np.take(arr, indices = np.array(index), axis = axis))

def find_flow_range(dx, dy, dz, mask):
    dx_valid = dx[mask]
    dy_valid = dy[mask]
    dz_valid = dz[mask]

    dx_min = dx_valid.min()
    dy_min = dy_valid.min()
    dz_min = dz_valid.min()

    dx_max = dx_valid.max()
    dy_max = dy_valid.max()
    dz_max = dz_valid.max()

    flow_min = np.min([dx_min, dy_min, dz_min])
    flow_max = np.max([dx_max, dy_max, dz_max])
    
    return flow_min, flow_max

'''
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
Program
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
'''
def analyze_at_volume(mask_np,
                      flow_volraft,
                      flow_gt,
                      flow_mbsvan,
                      flow_mbsopt,
                      logger):
    error_name = 'epe'
    def error(gt, measure, mask):
        # return rmse(gt, measure, mask)
        return epe(measure, gt, axis=0, mask=mask)

    error_volraft, _ = error(flow_gt, flow_volraft, mask_np)
    error_mbsvan, _ = error(flow_gt, flow_mbsvan, mask_np)
    error_mbsopt, _ = error(flow_gt, flow_mbsopt, mask_np)

    with logger:
        print(f'------------------------------------------------------')
        print(f'volume')
        print(f'{error_name} error mbs-van = {error_mbsvan}')
        print(f'{error_name} error mbs-opt = {error_mbsopt}')
        print(f'{error_name} error volraft = {error_volraft}')
        print(f'------------------------------------------------------')
    return

def error_at_slice(mask_np,
                   index,
                   axis,
                   flow_volraft,
                   flow_gt,
                   flow_mbsvan,
                   flow_mbsopt,
                   logger):
    error_name = 'epe'
    def error(gt, measure, mask):
        # return rmse(gt, measure, mask)
        return epe(measure, gt, axis=0, mask=mask)
    
    flow_gt_slice = np.take(flow_gt, indices = np.array(index), axis = axis+1)
    flow_mbsvan_slice = np.take(flow_mbsvan, indices = np.array(index), axis = axis+1)
    flow_mbsopt_slice = np.take(flow_mbsopt, indices = np.array(index), axis = axis+1)
    flow_volraft_slice = np.take(flow_volraft, indices = np.array(index), axis = axis+1)
    mask_slice = np.take(mask_np, indices = np.array(index), axis = axis)

    error_mbsvan, _ = error(flow_gt_slice, flow_mbsvan_slice, mask_slice)
    error_mbsopt, _ = error(flow_gt_slice, flow_mbsopt_slice, mask_slice)
    error_volraft, _ = error(flow_gt_slice, flow_volraft_slice, mask_slice)

    with logger:
        print(f'------------------------------------------------------')
        print(f'slice_axis = {axis}, slice = {index}')
        print(f'{error_name} error of slice mbs-van = {error_mbsvan}')
        print(f'{error_name} error of slice mbs-opt = {error_mbsopt}')
        print(f'{error_name} error of slice volraft = {error_volraft}')
        print(f'------------------------------------------------------')
    return

def analyze_bone_at_slice(logger, dataset, mask_bone, mask_screw, 
                          flow_mbsvan, flow_mbsopt, flow_volraft,
                          slice_axis, slice_axisname, slice_index,
                          has_gt_flow, flow_gt = None, default_flow_range = None):
    def print_range(arr):
        print(f'[{arr.min()}, {arr.max()}]')

    def slice_at(arr, index, axis, faxis, mask = None):
        if mask is None:
            mask = 1.0
        
        slice_arr = operation(np.take(arr[faxis, :, :, :], indices = np.array(index), axis = axis)) * mask
        return slice_arr

    # def roi(arr, slice_axis):
    #     # crop some region for x and y axis
    #     if slice_axis == 2:
    #         return arr[:, :]
    #     else:
    #         return arr[:-60, :]
        
    def roi(arr, bounding_box):
        # crop some region for x and y axis
        (min_row, min_col), (max_row, max_col) = bounding_box
        return arr[min_row:max_row+1, min_col:max_col+1]

    def find_bounding_box(binary_mask):
        # Find the indices of all "true" values in the binary mask
        true_positions = np.argwhere(binary_mask)
        
        # If there are no "true" values, return an indication of an empty bounding box
        if true_positions.size == 0:
            return None
        
        # Calculate min and max of the row and column indices
        min_row = np.min(true_positions[:, 0])
        max_row = np.max(true_positions[:, 0])
        min_col = np.min(true_positions[:, 1])
        max_col = np.max(true_positions[:, 1])
        
        # Return the top-left and bottom-right coordinates (row, col) of the bounding box
        return [(min_row, min_col), (max_row, max_col)]
    
    def plot_slice(plot_manager : PlotManager, 
                   image, 
                   screw_image, 
                   plot_range, 
                   each_figsize, 
                   slices_folder_path, 
                   prefix, 
                   slice_axisname, 
                   slice_index, 
                   cblabel = "Displacement (voxels)",
                   cmap = "viridis"):
        # Part 1: Save the image
        # Plot bone
        _, _ = plot_manager.plot_images(images = image, 
                                        titles = [""],
                                        limits = [plot_range],
                                        each_figsize = each_figsize,
                                        caxis_mode ='None',
                                        cmap = cmap)
        
        # Plot screw
        plt.imshow(screw_image, cmap='gray')
        plt.tight_layout()

        plt.savefig(FileHandler.join(slices_folder_path, f'{prefix}_{slice_axisname}{slice_index}.png'),
                    transparent = transparent, dpi = dpi, bbox_inches='tight', pad_inches=0)
        plt.savefig(FileHandler.join(slices_folder_path, f'{prefix}_{slice_axisname}{slice_index}.pdf'),
                    transparent = transparent, dpi = dpi, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Part 2: Save the colorbar only, more tightly and with a legend (text)
        fig, ax = plt.subplots(figsize=(0.5, each_figsize[1]))  # Adjusted for a tighter colorbar
        # Create a dummy image and use its color mapping to create the colorbar
        # Then, remove the dummy image
        norm = plt.Normalize(vmin = plot_range[0], vmax = plot_range[1])
        dummy_img = ax.imshow(image[0], cmap = cmap, norm = norm)
        plt.gca().set_visible(False)  # Hide the current axis

        # Creating the colorbar with specific aspect
        cbar = fig.colorbar(dummy_img, ax=ax, aspect=20, fraction=1.2, pad=0.04)
        cbar.ax.set_visible(True)  # Only make the colorbar visible

        # Adding text to the colorbar to serve as a legend
        cbar.set_label(cblabel, rotation=270, labelpad=15)

        plt.savefig(FileHandler.join(slices_folder_path, f'{prefix}_{slice_axisname}{slice_index}_cb.png'), 
                    transparent = transparent, dpi = dpi, bbox_inches='tight', pad_inches=0.02)
        plt.savefig(FileHandler.join(slices_folder_path, f'{prefix}_{slice_axisname}{slice_index}_cb.pdf'), 
                    transparent = transparent, dpi = dpi, bbox_inches='tight', pad_inches=0.02)
        plt.close()
        
        return
    
    transparent = True
    dpi = 300
    each_figsize = (6, 4)

    # Make subfolders for each slices
    slices_folder_path = FileHandler.join(logger.analysis_folder_path, f'comparison_{slice_axisname}{slice_index}')
    FileHandler.mkdir(slices_folder_path)

    plt.close()
    plot_manager = PlotManager()

    # store the gradient magnitude
    flow_mbsvan_gm = 0
    flow_mbsopt_gm = 0
    flow_volraft_gm = 0

    if has_gt_flow:
        flow_gt_gm = 0
        
        error_at_slice(mask_bone,
                       slice_index,
                       slice_axis,
                       flow_volraft,
                       flow_gt,
                       flow_mbsvan,
                       flow_mbsopt,
                       logger)

    bounding_box = None

    for slice_faxis in range(0, 3):
        # Plot flow mbsoptflow-optimal
        mask_bone_slice = operation(np.take(mask_bone, indices = np.array(slice_index), axis = slice_axis)).astype(bool)
        screw_slice = operation(np.take(mask_screw, indices = np.array(slice_index), axis = slice_axis)).astype(bool)

        # Get the bounding box of bone
        bounding_box = find_bounding_box(mask_bone_slice)

        if bounding_box is not None:
            vol0_slice = operation(np.take(dataset.vol0.data_np, indices = np.array(slice_index), axis = slice_axis))
            vol1_slice = operation(np.take(dataset.vol1.data_np, indices = np.array(slice_index), axis = slice_axis))

            if has_gt_flow:
                flow_gt_slice = roi(slice_at(flow_gt, index = slice_index, axis = slice_axis, faxis = slice_faxis, mask = mask_bone_slice), bounding_box)

            flow_mbsvan_slice = roi(slice_at(flow_mbsvan, index = slice_index, axis = slice_axis, faxis = slice_faxis, mask = mask_bone_slice), bounding_box)
            flow_mbsopt_slice = roi(slice_at(flow_mbsopt, index = slice_index, axis = slice_axis, faxis = slice_faxis, mask = mask_bone_slice), bounding_box)
            flow_volraft_slice = roi(slice_at(flow_volraft, index = slice_index, axis = slice_axis, faxis = slice_faxis, mask = mask_bone_slice), bounding_box)

            mask_bone_slice = roi(mask_bone_slice, bounding_box)
            vol0_slice = roi(vol0_slice, bounding_box)
            vol1_slice = roi(vol1_slice, bounding_box)
            screw_slice = roi(screw_slice, bounding_box)

            screw_image = np.ma.array(1-screw_slice, mask=~screw_slice)

            if has_gt_flow:
                flow_gt_gm += flow_gt_slice ** 2

            flow_mbsvan_gm += flow_mbsvan_slice ** 2
            flow_mbsopt_gm += flow_mbsopt_slice ** 2
            flow_volraft_gm += flow_volraft_slice ** 2

            if default_flow_range is None:
                if args.has_gt_flow:
                    # use the range of ground truth directly
                    flow_min, flow_max = find_percentiles([flow_gt_slice], mask_bone_slice.astype(bool),
                                                        percentiles=(0, 100))
                else:
                    flow_min, flow_max = find_percentiles([flow_mbsvan_slice, flow_mbsopt_slice, flow_volraft_slice], mask_bone_slice.astype(bool),
                                                        percentiles=(0, 100))
                flow_range = [flow_min, flow_max]
            else:
                flow_range = default_flow_range

            vol_min, vol_max = find_percentiles([vol0_slice, vol1_slice], mask_bone_slice.astype(bool),
                                                percentiles=(5, 95))
            vol_range = [vol_min, vol_max]

            plot_slice(plot_manager = plot_manager, image = [np.ma.array(vol0_slice, mask=~mask_bone_slice)],
                    screw_image = screw_image,
                    plot_range = vol_range, each_figsize = each_figsize, slices_folder_path = slices_folder_path,
                    prefix = "vol0", slice_axisname = slice_axisname, slice_index = slice_index, cmap = 'magma', cblabel='Intensity')
            
            plot_slice(plot_manager = plot_manager, image = [np.ma.array(vol1_slice, mask=~mask_bone_slice)],
                    screw_image = screw_image,
                    plot_range = vol_range, each_figsize = each_figsize, slices_folder_path = slices_folder_path,
                    prefix = "vol1", slice_axisname = slice_axisname, slice_index = slice_index, cmap = 'magma', cblabel='Intensity')
            
            if has_gt_flow:
                plot_slice(plot_manager = plot_manager, image = [np.ma.array(flow_gt_slice, mask=~mask_bone_slice)],
                            screw_image = screw_image,
                            plot_range = flow_range, each_figsize = each_figsize, slices_folder_path = slices_folder_path,
                            prefix = f"f{slice_faxis}_gt", slice_axisname = slice_axisname, slice_index = slice_index)
                
            plot_slice(plot_manager = plot_manager, image = [np.ma.array(flow_mbsvan_slice, mask=~mask_bone_slice)],
                            screw_image = screw_image,
                            plot_range = flow_range, each_figsize = each_figsize, slices_folder_path = slices_folder_path,
                            prefix = f"f{slice_faxis}_mbsvan", slice_axisname = slice_axisname, slice_index = slice_index)
            
            plot_slice(plot_manager = plot_manager, image = [np.ma.array(flow_mbsopt_slice, mask=~mask_bone_slice)],
                            screw_image = screw_image,
                            plot_range = flow_range, each_figsize = each_figsize, slices_folder_path = slices_folder_path,
                            prefix = f"f{slice_faxis}_mbsopt", slice_axisname = slice_axisname, slice_index = slice_index)
            
            plot_slice(plot_manager = plot_manager, image = [np.ma.array(flow_volraft_slice, mask=~mask_bone_slice)],
                            screw_image = screw_image,
                            plot_range = flow_range, each_figsize = each_figsize, slices_folder_path = slices_folder_path,
                            prefix = f"f{slice_faxis}_volraft", slice_axisname = slice_axisname, slice_index = slice_index)

    if bounding_box is not None:
        # Plot Gradient magnitude
        flow_mbsvan_gm = np.sqrt(flow_mbsvan_gm)
        flow_mbsopt_gm = np.sqrt(flow_mbsopt_gm)
        flow_volraft_gm = np.sqrt(flow_volraft_gm)

        if args.has_gt_flow:
            flow_gt_gm = np.sqrt(flow_gt_gm)

        if args.has_gt_flow:
            # use the range of ground truth directly
            flow_min, flow_max = find_percentiles([flow_gt_gm], mask_bone_slice.astype(bool),
                                                percentiles=(0, 100))
        else:
            flow_min, flow_max = find_percentiles([flow_mbsvan_gm, flow_mbsopt_gm, flow_volraft_gm], mask_bone_slice.astype(bool), percentiles=(0, 100))

        flow_range = [flow_min, flow_max]

        if args.has_gt_flow:
            plot_slice(plot_manager = plot_manager, image = [np.ma.array(flow_gt_gm, mask=~mask_bone_slice)],
                            screw_image = screw_image,
                            plot_range = flow_range, each_figsize = each_figsize, slices_folder_path = slices_folder_path,
                            prefix = f"fm_gt", slice_axisname = slice_axisname, slice_index = slice_index)


        plot_slice(plot_manager = plot_manager, image = [np.ma.array(flow_mbsvan_gm, mask=~mask_bone_slice)],
                    screw_image = screw_image,
                    plot_range = flow_range, each_figsize = each_figsize, slices_folder_path = slices_folder_path,
                    prefix = f"fm_mbsvan", slice_axisname = slice_axisname, slice_index = slice_index)

        plot_slice(plot_manager = plot_manager, image = [np.ma.array(flow_mbsopt_gm, mask=~mask_bone_slice)],
                    screw_image = screw_image,
                    plot_range = flow_range, each_figsize = each_figsize, slices_folder_path = slices_folder_path,
                    prefix = f"fm_mbsopt", slice_axisname = slice_axisname, slice_index = slice_index)

        plot_slice(plot_manager = plot_manager, image = [np.ma.array(flow_volraft_gm, mask=~mask_bone_slice)],
                    screw_image = screw_image,
                    plot_range = flow_range, each_figsize = each_figsize, slices_folder_path = slices_folder_path,
                    prefix = f"fm_volraft", slice_axisname = slice_axisname, slice_index = slice_index)

    plt.close()

    return

'''
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
Program
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
'''

def analysis(args):
    # Create objects
    if args.debug:
        logger = Logger(jobid = args.jobid, name = "debug_analysis", verbose = True)
    else:
        logger = Logger(jobid = args.jobid, name = "analysis")

    logger.build_output(output_folder_path = args.output_folder, 
                        make_checkpoint = False, 
                        make_result = False,
                        make_analysis = True,
                        dataset_name = args.dataset_name)
    logger.setup()

    if args.jobid is not None:
        print(f'JobID = {args.jobid}')

    # Read the version of packages
    print(f'torch version: {torch.__version__}')
    print(f'numpy version: {np.__version__}')
    print(f'matplotlib version: {mpl.__version__}')

    # Device setup
    device_controller = DeviceManager(verbose = False)  # Automatically select the best available device
    device = device_controller.get_device(use_cpu = args.cpu)

    # Data type setup
    ptdtype = torch.float32 # PyTorch data type
    npdtype = np.float32 # Numpy data type

    # Seed setup
    seed_manager = SeedManager(seed = args.seed)
    print(f"Seed: {seed_manager.get_current_seed()}")

    # Plotter setup
    mpl.use('Agg')
    print(f'matplotlib backend is: {mpl.get_backend()}')

    # Define dataset folder path
    vol0_path = FileHandler.join(args.dataset_path, "volume0")
    vol1_path = FileHandler.join(args.dataset_path, "volume1")
    mask_path = FileHandler.join(args.dataset_path, "mask")
    if args.has_gt_flow:
        flow_path = FileHandler.join(args.dataset_path, 'ground_truth', 'flow')
    else:
        flow_path = None

    dataset = DVCVolumeDataset(vol0_path = vol0_path, 
                               vol1_path = vol1_path,
                               mask_path = mask_path,
                               flow_path = flow_path)
    
    if len(dataset.mask.data.shape) == 3:
        nx, ny, nz = dataset.mask.data.shape
    else:
        _, nx, ny, nz = dataset.mask.data.shape

        # Squeeze the first channel of volumes
        dataset.vol0.data = np.squeeze(dataset.vol0.data)
        dataset.vol1.data = np.squeeze(dataset.vol1.data)
        dataset.mask.data = np.squeeze(dataset.mask.data)

        DeviceManager.garbage()

    # Convert volume0, volume1 and mask to xyz
    dataset.vol0.to_xyz()
    dataset.vol1.to_xyz()
    dataset.mask.to_xyz()

    if dataset.flow is not None:
        dataset.flow.to_xyz()

    # Load result from mbsoptflow-vanilla
    mbsvan = DVCFlow(npdtype = npdtype)
    mbsvan.load_data(folder_path = args.mbsvan_path)
    mbsvan.to_xyz()

    # Load result from mbsoptflow-optimal
    mbsopt = DVCFlow(npdtype = npdtype)
    mbsopt.load_data(folder_path = args.mbsopt_path)
    mbsopt.to_xyz()

    # Load result
    # list to store files
    res = []
    # Iterate directory
    for file in FileHandler.listdir(args.result_path):
        # check only text files
        if file.endswith('.npy'):
            res.append(file)
    result_path = FileHandler.join(args.result_path, res[-1])
    print(f'load result from: {result_path}')

    result = DVCFlow(flow_data = np.load(file = result_path), indexing = "dhw")
    result.to_xyz()
    print(f'result.data.shape = {result.data.shape}')

    # Load all matrix
    flow_volraft = result.data.copy()

    if args.has_gt_flow:
        flow_gt = dataset.flow.data.copy()
    else:
        flow_gt = None

    DeviceManager.garbage()
    
    # # Do a special operation to mbsoptflow result
    flow_mbsvan = np.zeros_like(mbsvan.data) # TODO: Find out why
    flow_mbsvan[0, :, :, :] = -1.0 * mbsvan.data[2, :, :, :] # TODO: Find out why
    flow_mbsvan[1, :, :, :] = -1.0 * mbsvan.data[0, :, :, :] # TODO: Find out why
    flow_mbsvan[2, :, :, :] = -1.0 * mbsvan.data[1, :, :, :] # TODO: Find out why

    # # Do a special operation to mbsoptflow result
    flow_mbsopt = np.zeros_like(mbsopt.data) # TODO: Find out why
    flow_mbsopt[0, :, :, :] = -1.0 * mbsopt.data[2, :, :, :] # TODO: Find out why
    flow_mbsopt[1, :, :, :] = -1.0 * mbsopt.data[0, :, :, :] # TODO: Find out why
    flow_mbsopt[2, :, :, :] = -1.0 * mbsopt.data[1, :, :, :] # TODO: Find out why    

    print(f'dataset.mask.shape = {dataset.mask.shape}, {dataset.mask.data.dtype}')
    print(f'flow_mbsvan.shape = {flow_mbsvan.shape}, {flow_mbsvan.dtype}')
    print(f'flow_mbsopt.shape = {flow_mbsopt.shape}, {flow_mbsopt.dtype}')
    print(f'flow_volraft.shape = {flow_volraft.shape}, {flow_volraft.dtype}')

    if args.has_gt_flow:
        print(f'flow_gt_mask.shape = {flow_gt.shape}, {flow_gt.dtype}')

    # Get original mask for degradation layer
    mask_watershed = DVCVolume()
    mask_watershed.load_data(folder_path = args.mask_watershed_path, tiff_subfolder="watershed")

    # Crop the sample holder
    if args.crop_zaxis < 0:
        dataset.vol0.data_np = dataset.vol0.data_np[:, :, :args.crop_zaxis]
        dataset.vol1.data_np = dataset.vol1.data_np[:, :, :args.crop_zaxis]
        dataset.mask.data_np = dataset.mask.data_np[:, :, :args.crop_zaxis]
        mask_watershed.data_np = mask_watershed.data_np[:, :, :args.crop_zaxis]
        flow_mbsvan = flow_mbsvan[:, :, :, :args.crop_zaxis]
        flow_mbsopt = flow_mbsopt[:, :, :, :args.crop_zaxis]
        flow_volraft = flow_volraft[:, :, :, :args.crop_zaxis]
        if args.has_gt_flow:
            flow_gt = flow_gt[:, :, :, :args.crop_zaxis]
    
    # Separate the mask
    DeviceManager.garbage()

    mask_bone = (mask_watershed.data_np == 1).astype(bool)
    mask_screw = (mask_watershed.data_np == 2).astype(bool)
    
    # Load json
    content = YAMLHandler.read_yaml(args.analysis_path)
    transparent = True
    dpi = 300
    each_figsize = (6, 4)

    if args.has_gt_flow:
        default_flow_range = [-24, 24]
    else:
        default_flow_range = None

    # Analysis at volume
    if args.has_gt_flow:
        analyze_at_volume(mask_np = mask_bone, 
                          flow_volraft = flow_volraft, 
                          flow_gt = flow_gt, 
                          flow_mbsvan = flow_mbsvan, 
                          flow_mbsopt = flow_mbsopt,
                          logger = logger)

    # Analysis at slices
    for idx, slice_task in enumerate(content["slice_tasks"]):
        if slice_task["enable"]:
            slice_index = slice_task["slice_index"]
            slice_axis = slice_task["slice_axis"]
            slice_axisname = slice_task["slice_axisname"]

            analyze_bone_at_slice(logger, dataset, mask_bone, mask_screw, 
                          flow_mbsvan, flow_mbsopt, flow_volraft,
                          slice_axis, slice_axisname, slice_index,
                          args.has_gt_flow, flow_gt = flow_gt, default_flow_range = default_flow_range)

    # Final return
    logger.teardown()

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
    parser = argparse.ArgumentParser(description='PyTorch Neural Network Training')
    parser.add_argument('analysis_path', type=str, help='File path to the YAML file of analysis tasks')
    parser.add_argument('mask_watershed_path', type=str, help="Folder path to the original watershed folder")
    parser.add_argument('dataset_path', type=str, help='Folder path to the dataset')
    parser.add_argument('result_path', type=str, help='Folder path to the result')
    parser.add_argument('mbsvan_path', type=str, help='Folder path to the mbsoptflow-vanilla folder')
    parser.add_argument('mbsopt_path', type=str, help='Folder path to the mbsoptflow-optimal folder')
    parser.add_argument('dataset_name', type=str, help='Name of the dataset')
    parser.add_argument('--has_gt_flow', type=int, default=1, help='Indicate the dataset has the ground truth flow')
    parser.add_argument('--jobid', type=str, default=None, help='String that represent the job')
    parser.add_argument('--output_folder', type=str, default='./output', help='Folder path of the output')
    parser.add_argument('--seed', type=int, default=None, help='Seed for random number, None for random')
    parser.add_argument('--slice_axis', type=int, default=0, help='Axis index of slice (0 for x, 1 for y, 2 for z)')
    parser.add_argument('--slice_axisname', type=str, default='x', help='String for the slice axis')
    parser.add_argument('--slice_index', type=int, default=0, help='Index value of slice')
    parser.add_argument('--crop_zaxis', type=int, default=-60, help='Cropping the sample holder.')
    parser.add_argument('--max_flow', type=float, default=None, help='Maximum range of flow field')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for figures')
    parser.add_argument('--transparent', type=bool, default=True, help='Transparent background for figures')
    parser.add_argument('--cpu', action='store_true', help='Enable CPU for training')
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    args = parser.parse_args()

    # Training the model
    analysis(args)

    ## For testing purpose


import argparse
import torch
import torch.optim as optim
import numpy as np
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
def normalize_data_by_mask(data_np, mask_np):
    # Applying the mask to select only valid elements
    valid_data = data_np[mask_np]

    # Normalizing only valid elements to range [0, 1]
    data_min = valid_data.min()
    data_max = valid_data.max()

    # return the normalized range
    return (data_np - data_min) / (data_max - data_min)


def generate_gaussian_3d(shape, sigma):
    # Generate the grid of (x, y, z) coordinates
    m, n, k = [(ss - 1.) / 2. for ss in shape]
    y, x, z = np.ogrid[-m:m+1, -n:n+1, -k:k+1]

    # Calculate the Gaussian matrix
    h = np.exp(-(x**2 / (2. * sigma[0]**2) + y**2 / (2. * sigma[1]**2) + z**2 / (2. * sigma[2]**2)))

    return h / h.sum()

def generate_windows_3d(shape, margin = (0, 0, 0)):
    batch_size, _, NX, NY, NZ = shape

    # # Prepare hamming windows
    # hamming_x = np.zeros(shape = (NX), dtype = np.float32)
    # hamming_y = np.zeros(shape = (NY), dtype = np.float32)
    # hamming_z = np.zeros(shape = (NZ), dtype = np.float32)

    # # Generate 1D Hamming windows for each dimension
    # if margin[0] > 0:
    #     hamming_x[margin[0]//2:margin[0]//-2] = np.hamming(NX - margin[0])
    # else:
    #     hamming_x = np.hamming(NX)
    
    # if margin[1] > 0:
    #     hamming_y[margin[1]//2:margin[1]//-2] = np.hamming(NY - margin[1])
    # else:
    #     hamming_y = np.hamming(NY)

    # if margin[2] > 0:
    #     hamming_z[margin[2]//2:margin[2]//-2] = np.hamming(NZ - margin[2])
    # else:
    #     hamming_z = np.hamming(NZ)

    # # Extend to 3D by broadcasting
    # window_3d = hamming_x[:, np.newaxis, np.newaxis] * hamming_y[np.newaxis, :, np.newaxis] * hamming_z[np.newaxis, np.newaxis, :]
    
    # Use Gaussian as the kernel
    min_dim = np.min([NX, NY, NZ])
    window_3d = generate_gaussian_3d(shape = (NX, NY, NZ),
                                     sigma = (min_dim / 6.0, min_dim / 6.0, min_dim / 6.0))

    # Add batch and singleton dimensions
    # Using np.newaxis to add the required dimensions
    window_batch = np.expand_dims(window_3d, axis=0)  # Add batch dimension
    window_batch = np.expand_dims(window_batch, axis=1)  # Add singleton dimension

    # Repeat the tensor for the whole batch
    window_batch = np.repeat(window_batch, batch_size, axis=0)

    # hamming_3d now contains the 3D Hamming window weights of size
    return window_batch, window_3d

def save_figures(dataset, logger, flow_infer, slice_index, slice_axis, slice_axisname, flow_range=None):
    plt.close('all')

    plotter = PlotManager()

    # slice_index = 640
    # slice_axis = 1 # 0 for x, 1 for y, 2 for z
    # slice_axisname = 'y'

    if dataset.flow is None:
        flow_slice_gt = None
    else:
        flow_slice_gt = np.take(dataset.flow.data_np, np.array(slice_index), slice_axis + 1)

    flow_slice_infer = np.take(flow_infer, np.array(slice_index), slice_axis + 1)
    mask_slice = np.take(dataset.mask.data_np, np.array(slice_index), slice_axis).astype(bool)

    dx_infer = flow_slice_infer[0, :, :]
    dy_infer = flow_slice_infer[1, :, :]
    dz_infer = flow_slice_infer[2, :, :]

    if flow_slice_gt is None:
        dx_gt = dy_gt = dz_gt = None
    else:
        dx_gt = flow_slice_gt[0, :, :]
        dy_gt = flow_slice_gt[1, :, :]
        dz_gt = flow_slice_gt[2, :, :]

    if flow_range is None:
        if flow_slice_gt is None:
            flow_range = (flow_slice_infer.min(), flow_slice_gt.max())
        else:
            flow_range = (flow_slice_gt.min(), flow_slice_gt.max())

    if dx_gt is not None:
        _, _ = plotter.plot_images(images = [np.ma.array(dx_gt, mask=~mask_slice), np.ma.array(dx_infer, mask=~mask_slice)], 
                                   titles = [f'dx gt at {slice_axisname}={slice_index}', f'dx infer at {slice_axisname}={slice_index}'],
                                   limits = [flow_range, flow_range, flow_range],
                                   each_figsize = (6, 4),
                                   caxis_mode='v')
    else:
        _, _ = plotter.plot_images(images = [np.ma.array(dx_infer, mask=~mask_slice)], 
                                   titles = [f'dx infer at {slice_axisname}={slice_index}'],
                                   limits = [flow_range],
                                   each_figsize = (6, 4),
                                   caxis_mode='v')

    plt.savefig(FileHandler.join(logger.result_folder_path, f'inference_flow_{slice_axisname}{slice_index}_dx.png'), transparent = True, dpi = 150)

    if dy_gt is not None:
        _, _ = plotter.plot_images(images = [np.ma.array(dy_gt, mask=~mask_slice), np.ma.array(dy_infer, mask=~mask_slice)], 
                                   titles = [f'dy gt at {slice_axisname}={slice_index}', f'dy infer at {slice_axisname}={slice_index}'],
                                   limits = [flow_range, flow_range, flow_range],
                                   each_figsize = (6, 4),
                                   caxis_mode='v')
    else:
        _, _ = plotter.plot_images(images = [np.ma.array(dy_infer, mask=~mask_slice)], 
                                   titles = [f'dy infer at {slice_axisname}={slice_index}'],
                                   limits = [flow_range],
                                   each_figsize = (6, 4),
                                   caxis_mode='v')

    plt.savefig(FileHandler.join(logger.result_folder_path, f'inference_flow_{slice_axisname}{slice_index}_dy.png'), transparent = True, dpi = 150)
    
    if dz_gt is not None:
        _, _ = plotter.plot_images(images = [np.ma.array(dz_gt, mask=~mask_slice), np.ma.array(dz_infer, mask=~mask_slice)], 
                                   titles = [f'dz gt at {slice_axisname}={slice_index}', f'dz infer at {slice_axisname}={slice_index}'],
                                   limits = [flow_range, flow_range, flow_range],
                                   each_figsize = (6, 4),
                                   caxis_mode='v')
    else:
        _, _ = plotter.plot_images(images = [np.ma.array(dz_infer, mask=~mask_slice)], 
                                   titles = [f'dz infer at {slice_axisname}={slice_index}'],
                                   limits = [flow_range],
                                   each_figsize = (6, 4),
                                   caxis_mode='v')

    plt.savefig(FileHandler.join(logger.result_folder_path, f'inference_flow_{slice_axisname}{slice_index}_dz.png'), transparent = True, dpi = 150)

    return

'''
# This method is inferring for each position (x,y,z) using neighboring patch
# And assign to the center (fx, fy, fz) position for the flow
# Suitable for inference of targeted flow with a certain size
'''
def split_range(start, end, step, num_splits):
    """
    Splits a range into multiple parts.

    :param start: Start of the range.
    :param end: End of the range.
    :param step: Step size of the range.
    :param num_splits: Number of parts to split the range into.
    :return: A list of tuples, each representing the start and end of a split.
    """
    total_elements = (end - start + step - 1) // step
    elements_per_split = total_elements // num_splits
    extra_elements = total_elements % num_splits

    splits = []
    current_start = start

    for i in range(num_splits):
        current_end = current_start + elements_per_split * step
        # Distribute extra elements among the first few splits
        if i < extra_elements:
            current_end += step

        if current_end > end:
            current_end = end

        splits.append((current_start, current_end))
        current_start = current_end

    return splits

'''
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
Program
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
'''
def print_versions() -> None:
    print(f'torch version: {torch.__version__}')
    print(f'numpy version: {np.__version__}')
    print(f'matplotlib version: {mpl.__version__}')

def setup_environment(args):
    """
    Setup the training environment based on the provided arguments
    """
    # Setup Logger
    logger = Logger(jobid=args.jobid, name='debug_infer' if args.debug else 'infer', verbose=args.debug)

    logger.build_output(output_folder_path = args.output_folder,
                        make_checkpoint = False,
                        make_result = True,
                        make_analysis = False,
                        dataset_name = args.dataset_name)
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

    # Define data type
    ptdtype = torch.float32 # PyTorch data type
    npdtype = np.float32 # Numpy data type

    # Setup matplotlib
    mpl.use('Agg')
    print(f'matplotlib backend is: {mpl.get_backend()}')

    return logger, device_manager, device, ptdtype, npdtype

def inference_on_device(device_id, model,
                        batch_size, flow_pad_infer, shifts,
                        weights_pad_infer, windows_batch, windows_3d,
                        posx_start_begin, posx_start_end,
                        nx_pad, ny_pad, nz_pad,
                        patch_shape, flow_shape, flow_margin_target, 
                        mask_pad, vol0_pad, vol1_pad, 
                        npdtype, ptdtype, 
                        status_dict, verbose=False):
    
    # Set default status as fail
    status_dict[device_id] = 'Failed'

    # Try
    try:
        batch_vol0_np = np.zeros(shape = (batch_size, 1, patch_shape[2], patch_shape[3], patch_shape[4]), dtype=npdtype)
        batch_vol1_np = np.zeros(shape = (batch_size, 1, patch_shape[2], patch_shape[3], patch_shape[4]), dtype=npdtype)
        batch_posx = np.zeros(shape = (batch_size, 2), dtype=np.int32)
        batch_posy = np.zeros(shape = (batch_size, 2), dtype=np.int32)
        batch_posz = np.zeros(shape = (batch_size, 2), dtype=np.int32)

        batch_count = 0
        
        for shift_idx in range(shifts.shape[0]):
            # Get shift
            shift = shifts[shift_idx, :]
            print(f'shift {shift_idx} = {shift}')

            full_margin = np.array(args.full_margin)
            half_margin = full_margin // 2
            posx_stride = flow_shape[2] - full_margin[0]
            posy_stride = flow_shape[3] - full_margin[1]
            posz_stride = flow_shape[4] - full_margin[2]
            print(f'strides = [{posx_stride}, {posy_stride}, {posz_stride}]')

            for posx_start in range(posx_start_begin + shift[0], posx_start_end, posx_stride):
                for posy_start in range(patch_shape[3] + shift[1], ny_pad - patch_shape[3] + 1, posy_stride):
                    for posz_start in range(patch_shape[4] + shift[2], nz_pad - patch_shape[4] + 1, posz_stride):
                        posx_end = posx_start + flow_shape[2]
                        posy_end = posy_start + flow_shape[3]
                        posz_end = posz_start + flow_shape[4]
                        
                        # slice_x = slice(posx_start, posx_end)
                        # slice_y = slice(posy_start, posy_end)
                        # slice_z = slice(posz_start, posz_end)

                        slice_x = slice(posx_start + half_margin[0], posx_end - half_margin[0])
                        slice_y = slice(posy_start + half_margin[1], posy_end - half_margin[1])
                        slice_z = slice(posz_start + half_margin[2], posz_end - half_margin[2])

                        patch_mask_flow = mask_pad[slice_x, slice_y, slice_z]

                        if np.any(patch_mask_flow):
                            margin_begin = flow_margin_target[0]
                            margin_end = flow_margin_target[1]
                            vol_slice_x = slice(posx_start - margin_begin[0], posx_end + margin_end[0])
                            vol_slice_y = slice(posy_start - margin_begin[1], posy_end + margin_end[1])
                            vol_slice_z = slice(posz_start - margin_begin[2], posz_end + margin_end[2])

                            batch_vol0_np[batch_count, 0, :, :, :] = vol0_pad[vol_slice_x, vol_slice_y, vol_slice_z]
                            batch_vol1_np[batch_count, 0, :, :, :] = vol1_pad[vol_slice_x, vol_slice_y, vol_slice_z]
                            batch_posx[batch_count, :] = [posx_start, posx_end]
                            batch_posy[batch_count, :] = [posy_start, posy_end]
                            batch_posz[batch_count, :] = [posz_start, posz_end]

                            batch_count += 1

                            # Do inference when the batch is filled up
                            if batch_count >= batch_size:
                                # Inference
                                device = torch.device(f'cuda:{device_id}')
                                batch_vol0 = torch.from_numpy(batch_vol0_np).to(device, dtype = ptdtype, non_blocking = True)
                                batch_vol1 = torch.from_numpy(batch_vol1_np).to(device, dtype = ptdtype, non_blocking = True)
                                windows_tensor = torch.from_numpy(windows_batch).to(device, dtype = ptdtype, non_blocking = True)
                                batch_flow = model.forward(batch_vol0, batch_vol1)

                                # Assign the flow to the batch
                                # Determine if the forward variable is a list
                                if isinstance(batch_flow, list):
                                    batch_flow_weighted = batch_flow[-1] * windows_tensor

                                    # Get the last batch
                                    batch_flow_np = batch_flow_weighted.cpu().detach().numpy()
                                else:
                                    batch_flow_weighted = batch_flow * windows_tensor

                                    batch_flow_np = batch_flow_weighted.cpu().detach().numpy()

                                if verbose:
                                    print(f'device {device_id}: inference {batch_count}')

                                for idx in range(batch_count):
                                    # assign the predicted values to flow
                                    # for margin = 0
                                    # batch_slice_x = slice(batch_posx[idx, 0], batch_posx[idx, 1])
                                    # batch_slice_y = slice(batch_posy[idx, 0], batch_posy[idx, 1])
                                    # batch_slice_z = slice(batch_posz[idx, 0], batch_posz[idx, 1])
                                    # flow_pad_infer[:, batch_slice_x, batch_slice_y, batch_slice_z, shift_idx] = batch_flow_np[idx, :, :, :, :]

                                    # for margin not 0
                                    batch_slice_x = slice(batch_posx[idx, 0] + half_margin[0], batch_posx[idx, 1] - half_margin[0])
                                    batch_slice_y = slice(batch_posy[idx, 0] + half_margin[1], batch_posy[idx, 1] - half_margin[1])
                                    batch_slice_z = slice(batch_posz[idx, 0] + half_margin[2], batch_posz[idx, 1] - half_margin[2])
                                    predict_slice_x = slice(half_margin[0], flow_shape[2] - half_margin[0])
                                    predict_slice_y = slice(half_margin[1], flow_shape[3] - half_margin[1])
                                    predict_slice_z = slice(half_margin[2], flow_shape[4] - half_margin[2])
                                    # flow_pad_infer[:, batch_slice_x, batch_slice_y, batch_slice_z, shift_idx] = batch_flow_np[idx, :, predict_slice_x, predict_slice_y, predict_slice_z]

                                    # Compute the weighted flow
                                    # batch_flow_np_weighted = batch_flow_np * windows_batch
                                    flow_pad_infer[:, batch_slice_x, batch_slice_y, batch_slice_z, shift_idx] = batch_flow_np[idx, :, predict_slice_x, predict_slice_y, predict_slice_z]

                                    weights_pad_infer[:, batch_slice_x, batch_slice_y, batch_slice_z, shift_idx] = windows_batch[idx, :, predict_slice_x, predict_slice_y, predict_slice_z]

                                # Reset batch
                                batch_count = 0
                                batch_vol0_np.fill(0.0)
                                batch_vol1_np.fill(0.0)
                                batch_posx.fill(0)
                                batch_posy.fill(0)
                                batch_posz.fill(0)

                                # Clear memory
                                batch_vol0 = None
                                batch_vol1 = None
                                batch_flow = None
                                windows_tensor = None
                                batch_flow_weighted = None
                                DeviceManager.garbage()

            # The final inference for the last incompleted batch
            device = torch.device(f'cuda:{device_id}')
            if verbose:
                print(f'device {device_id}: last inference {batch_count}')

            batch_vol0 = torch.from_numpy(batch_vol0_np).to(device, dtype = ptdtype, non_blocking = True)
            batch_vol1 = torch.from_numpy(batch_vol1_np).to(device, dtype = ptdtype, non_blocking = True)
            windows_tensor = torch.from_numpy(windows_batch).to(device, dtype = ptdtype, non_blocking = True)
            batch_flow = model.forward(batch_vol0, batch_vol1)

            # Assign the flow to the batch
            if isinstance(batch_flow, list):
                batch_flow_weighted = batch_flow[-1] * windows_tensor

                # Get the last batch
                batch_flow_np = batch_flow_weighted.cpu().detach().numpy()
            else:
                batch_flow_weighted = batch_flow * windows_tensor

                batch_flow_np = batch_flow_weighted.cpu().detach().numpy()

            for idx in range(batch_count):
                # assign the predicted values to flow
                # for margin = 0
                # batch_slice_x = slice(batch_posx[idx, 0], batch_posx[idx, 1])
                # batch_slice_y = slice(batch_posy[idx, 0], batch_posy[idx, 1])
                # batch_slice_z = slice(batch_posz[idx, 0], batch_posz[idx, 1])
                # flow_pad_infer[:, batch_slice_x, batch_slice_y, batch_slice_z, shift_idx] = batch_flow_np[idx, :, :, :, :]

                # for margin not 0
                batch_slice_x = slice(batch_posx[idx, 0] + half_margin[0], batch_posx[idx, 1] - half_margin[0])
                batch_slice_y = slice(batch_posy[idx, 0] + half_margin[1], batch_posy[idx, 1] - half_margin[1])
                batch_slice_z = slice(batch_posz[idx, 0] + half_margin[2], batch_posz[idx, 1] - half_margin[2])
                predict_slice_x = slice(half_margin[0], flow_shape[2] - half_margin[0])
                predict_slice_y = slice(half_margin[1], flow_shape[3] - half_margin[1])
                predict_slice_z = slice(half_margin[2], flow_shape[4] - half_margin[2])
                # flow_pad_infer[:, batch_slice_x, batch_slice_y, batch_slice_z, shift_idx] = batch_flow_np[idx, :, predict_slice_x, predict_slice_y, predict_slice_z]

                # Compute the weighted flow
                flow_pad_infer[:, batch_slice_x, batch_slice_y, batch_slice_z, shift_idx] = batch_flow_np[idx, :, predict_slice_x, predict_slice_y, predict_slice_z]

                weights_pad_infer[:, batch_slice_x, batch_slice_y, batch_slice_z, shift_idx] = windows_batch[idx, :, predict_slice_x, predict_slice_y, predict_slice_z]

            # Reset batch
            batch_count = 0
            batch_vol0_np.fill(0.0)
            batch_vol1_np.fill(0.0)
            batch_posx.fill(0)
            batch_posy.fill(0)
            batch_posz.fill(0)

            # Clear memory
            batch_vol0 = None
            batch_vol1 = None
            batch_flow = None
            windows_tensor = None
            batch_flow_weighted = None
            DeviceManager.garbage()

        # Update status as success if no error occurs
        status_dict[device_id] = 'Success'

    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            status_dict[device_id] = 'CUDA OOM Error'
        else:
            status_dict[device_id] = str(e)
            raise e
            
    # except Exception as e:
    #     # Handle other exceptions if necessary
    #     status_dict[device_id] = f'Failed with exception: {str(e)}'

    return

def infer(args):
    '''
    Infer
    '''
    # Setup environment
    logger, device_manager, device, ptdtype, npdtype = setup_environment(args)

    # Make sure that no gradient is generated
    with torch.no_grad():
        # Load checkpoint
        checkpoint_controller = CheckpointController(checkpoint_dir = args.checkpoint_path)

        # Load checkpoint for the details
        epoch, patch_shape, flow_shape, \
            _, _, _, \
            loss_list_train, loss_list_valid = \
                checkpoint_controller.load_last_checkpoint(network = None, 
                                                        optimizer = None, 
                                                        scheduler = None)
                
        # Load config
        config_path = checkpoint_controller.find_config_file()
        config = YAMLHandler.read_yaml(config_path)
        print(f'load configuration from {config_path}')

        # Setup seed
        seed_manager = SeedManager(seed = config["seed"])
        print(f"Seed: {seed_manager.get_current_seed()}")
        
        # Define network
        model  = ModelFactory.build_instance(patch_shape = patch_shape,
                                             flow_shape = flow_shape,
                                             config = config,
                                             ptdtype = ptdtype)
        
        # Load network weights from checkpoint
        _, _, _, model, _, _, _, _ = \
                checkpoint_controller.load_last_checkpoint(network = model, 
                                                        optimizer = None, 
                                                        scheduler = None)
        
        # Load to device
        device = device_manager.get_device()
        model.to(device)

        # Define dataset folder path
        vol0_path = FileHandler.join(args.dataset_path, "volume0")
        vol1_path = FileHandler.join(args.dataset_path, "volume1")
        mask_path = FileHandler.join(args.dataset_path, "mask")

        # Load dataset
        dataset = DVCVolumeDataset(vol0_path = vol0_path, 
                                vol1_path = vol1_path,
                                mask_path = mask_path,
                                flow_path = None)
        dataset.vol0.to_dhw()
        dataset.vol1.to_dhw()
        dataset.mask.to_dhw()

        print(f'vol0.shape = {dataset.vol0.shape}')
        print(f'vol1.shape = {dataset.vol1.shape}')
        print(f'mask.shape = {dataset.mask.shape}')

        # Correct the channel dimension
        if len(dataset.mask.data_np.shape) == 3:
            nx, ny, nz = dataset.mask.data_np.shape
        else:
            _, nx, ny, nz = dataset.mask.data_np.shape

            # Squeeze the first channel of volumes
            dataset.vol0.data_np = np.squeeze(dataset.vol0.data_np)
            dataset.vol1.data_np = np.squeeze(dataset.vol1.data_np)
            dataset.mask.data_np = np.squeeze(dataset.mask.data_np)

            DeviceManager.garbage()

        # Pad zeros to all volumes, mask and flows
        pad_width = ((patch_shape[2], patch_shape[2]),
                    (patch_shape[3], patch_shape[3]),
                    (patch_shape[4], patch_shape[4]))
        vol0_pad = np.pad(dataset.vol0.data_np, pad_width = pad_width, 
                        mode = 'constant', constant_values = 0)
        vol1_pad = np.pad(dataset.vol1.data_np, pad_width = pad_width,
                        mode = 'constant', constant_values = 0)
        mask_pad = np.pad(dataset.mask.data_np, pad_width = pad_width,
                        mode = 'constant', constant_values = 0).astype(bool)
        
        print(f'vol0_pad.shape = {vol0_pad.shape}')
        print(f'vol1_pad.shape = {vol1_pad.shape}')
        print(f'mask_pad.shape = {mask_pad.shape}')

        # Get patch size
        # patch_size = (patch_shape[2], patch_shape[3], patch_shape[4])

        # Prepare stride in (x, y, z)
        num_pixel = np.array([flow_shape[2:]])
        num_overlaps = np.minimum(args.num_overlaps, np.min(num_pixel))
        stride = num_pixel // num_overlaps
        print(f'stride = {stride}')

        # Create an array from -half_range to +half_range
        half_range = num_overlaps // 2
        shift_values = np.arange(-half_range, half_range + 1)

        # Transpose the array to make it a column vector and multiply
        shifts = np.matmul(shift_values.reshape(-1, 1), stride)
        print(f'shifts.shape = {shifts.shape} \n {shifts}')

        # Define flow after inference
        flow_pad_infer = np.zeros(shape = (3, mask_pad.shape[0], mask_pad.shape[1], mask_pad.shape[2], num_overlaps), 
                                  dtype = npdtype)
        print(f'flow_pad_infer.shape = {flow_pad_infer.shape}')
        print(f'patch_shape = {patch_shape}')

        flow_center = [(flow_shape[2] - 1) // 2,
                       (flow_shape[3] - 1) // 2,
                       (flow_shape[4] - 1) // 2]
        print(f'flow_shape = {flow_shape}')
        print(f'flow_center = {flow_center}')

        # Get the margin of flow
        flow_margin_target = [
            ((patch_shape[2] - flow_shape[2] + 1) // 2, 
            (patch_shape[3] - flow_shape[3] + 1) // 2, 
            (patch_shape[4] - flow_shape[4] + 1) // 2), 
            ((patch_shape[2] - flow_shape[2]) // 2, 
            (patch_shape[3] - flow_shape[3]) // 2, 
            (patch_shape[4] - flow_shape[4]) // 2)
            ]
        print(f'flow_margin_target = {flow_margin_target}')

        default_batch_size = args.infer_batch_size

        # Start evaluation of the network
        model.eval()

        # loop for each start of position index
        nx_pad, ny_pad, nz_pad = mask_pad.shape

        batch_size = default_batch_size
        print(f'batch_size = {batch_size}')
        print(f'full_margin = {args.full_margin}')

        # Generate Hamming windows
        windows_batch, windows_3d = generate_windows_3d(shape = (batch_size, 1, patch_shape[2], patch_shape[3], patch_shape[4]),
                                                        margin = args.full_margin)
        weights_pad_infer = np.zeros(shape = (3, mask_pad.shape[0], mask_pad.shape[1], mask_pad.shape[2], num_overlaps), 
                                     dtype = npdtype)
        print(f'windows_batch.shape = {windows_batch.shape}')
        print(f'windows_3d.shape = {windows_3d.shape}')
        print(f'weights_pad_infer.shape = {weights_pad_infer.shape}')

        # Prepare the timer to record inference time
        timer = Timer('Inference')
        timer.start()
        
        # For single GPU
        status_dict = dict()
        verbose = args.debug

        device_id = device_manager.get_device_id()
        posx_start_begin = patch_shape[2]
        posx_start_end = nx_pad - patch_shape[2] + 1
        logger.teardown()
        inference_on_device(device_id, model,
                            batch_size, flow_pad_infer, shifts,
                            weights_pad_infer, windows_batch, windows_3d,
                            posx_start_begin, posx_start_end,
                            nx_pad, ny_pad, nz_pad,
                            patch_shape, flow_shape, flow_margin_target, 
                            mask_pad, vol0_pad, vol1_pad, 
                            npdtype, ptdtype,
                            status_dict, verbose)
        logger.setup()

        # Check if all processes finished successfully
        all_successful = all(status == 'Success' for status in status_dict.values())
        if all_successful:
            print('All processes finished successfully')
        else:
            for device_id, status in status_dict.items():
                print(f'Process {device_id} status: {status}')
            
        if all_successful:
            # Remove the padding from flow
            idx_x = slice(patch_shape[2], patch_shape[2] + nx)
            idx_y = slice(patch_shape[3], patch_shape[3] + ny)
            idx_z = slice(patch_shape[4], patch_shape[4] + nz)

            # Approach 1: Find the median of inferred flow
            # flow_infer = np.median(flow_pad_infer[:, idx_x, idx_y, idx_z, :], axis = 4)

            # # Approach 2: Find the mean of inferred flow
            # flow_infer = np.mean(flow_pad_infer[:, idx_x, idx_y, idx_z, :], axis = 4)

            # # Approach 3: Sort along the last axis and exclude the top-1 and bottom-1 elements
            # sorted_flow = np.sort(flow_pad_infer[:, idx_x, idx_y, idx_z, :], axis = 4)
            # trimmed_flow = sorted_flow[..., 1:-1]  # Excludes the first and last element in the last dimension
            # flow_infer = np.mean(trimmed_flow, axis=4) # Compute the mean of the remaining elements
            
            # Approach 4: Compute by the weights
            weights_infer = np.sum(weights_pad_infer[:, idx_x, idx_y, idx_z, :], axis = 4)
            flow_infer = np.sum(flow_pad_infer[:, idx_x, idx_y, idx_z, :], axis = 4)
            flow_infer[weights_infer > 0.0] /= weights_infer[weights_infer > 0.0]

            # Normalize the flow by the weights
            # weights_infer = weights_pad_infer[:, idx_x, idx_y, idx_z, 0]
            # flow_infer[weights_infer > 0.0] /= weights_infer[weights_infer > 0.0]

            # Apply the mask (vectorized operation)
            mask_to_apply = mask_pad[idx_x, idx_y, idx_z]
            flow_infer *= mask_to_apply[None, :, :, :]  # Broadcasting mask across the first dimension

            print(f'flow_infer.shape = {flow_infer.shape}')

            # Conver to dhw
            # flow_infer = np.transpose(flow_infer, axes = (0, 3, 2, 1))
            # print(f'flow_infer.shape = {flow_infer.shape} after transpose')
            
            # Stop the timer
            timer.stop()
            elapsed = timer.elapsed_time()

            print(f'Inference time (seconds): {elapsed}')

            # Save to result
            result_path = FileHandler.join(logger.result_folder_path, f'result_{logger.time_stamp}.npy')
            print(f'Save result to {result_path}')
            np.save(result_path, flow_infer)

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
    parser.add_argument('dataset_path', type=str, help='Folder path to the dataset')
    # parser.add_argument('dataset_path', type=str, help='File path to the JSON file of dataset')
    parser.add_argument('checkpoint_path', type=str, help='Folder path to the checkpoint')
    parser.add_argument('dataset_name', type=str, help='Name of the dataset')
    parser.add_argument('--jobid', type=str, default=None, help='String that represent the job')
    parser.add_argument('--output_folder', type=str, default='./output', help='Folder path of the output')
    parser.add_argument('--full_margin', type=int, nargs='+', default=[20, 40, 40], help='The target patch size in (X, Y, Z) format')
    parser.add_argument('--infer_batch_size', type=int, default=500, help='Batch size for inference (default 500 for 32GB GPU memory)')
    parser.add_argument('--num_overlaps', type=int, default=5, help='Number of overlaps')
    # parser.add_argument('--num_gpus', type=int, default=0, help='Number of GPU')
    parser.add_argument('--cpu', action='store_true', help='Enable CPU for training')
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    args = parser.parse_args()

    # Training the model
    infer(args)
   
    # end of main function
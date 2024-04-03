import torch
import torch.nn.functional as F
import numpy as np

class VolumeWarping:
    @staticmethod
    def normalize(x, value_min=None, value_max=None, clip=None):
        '''
        To normalize a tensor/numpy array by the min-max values and by a clip limit
        '''
        if clip is None:
            if value_min is None and value_max is None:
                if (torch.is_tensor(x)):
                    return (x - torch.min(x)) / (torch.max(x) - torch.min(x))
                else:
                    return (x - np.min(x)) / (np.max(x) - np.min(x))
            else:
                return (x - value_min) / (value_max - value_min)
        elif torch.is_tensor(x):
            return torch.clamp((x - value_min) / (value_max - value_min), 
                            min=clip[0], 
                            max=clip[1])
        else:
            return np.clip((x - value_min) / (value_max - value_min), 
                        a_min=clip[0], 
                        a_max=clip[1])
            
    @staticmethod
    def get_flow_grid_new(shape, norm_range = None):
        '''
        Build a tensor that represent the flow field
        (0, 0) is at center
        '''
        # shape: tuple, can be 2D or 3D
        vectors = list()

        # for each dimension
        for idx, shape_dim in enumerate(shape):
            # Get the linear vector first
            vector = torch.linspace(start = 0, end = shape_dim - 1, steps = shape_dim)

            # Find the center coordinate of this dimension
            center = (shape_dim - 1) / 2

            # Compute the linear vector centered at the middle
            vector = vector - center

            # (optional) Normalize to the specific range
            if norm_range is not None:
                # Normalize vector to [0, 1]
                vector = VolumeWarping.normalize(vector)

                # Map to the range [min, max]
                vector = vector * (norm_range[1] - norm_range[0]) + norm_range[0]

            # Stack the vectors
            vectors.append(vector)

        # Compute the flow grid
        flow_grid = torch.meshgrid(vectors, indexing='ij')

        # Stack the flow grid into a single matrix
        flow_grid = torch.stack(flow_grid, dim = len(shape))

        # Flip the flow
        # flow_grid = torch.permute(flow_grid, dims = (1, 0, 2))
        
        return flow_grid

    @staticmethod
    def warp(vol0, flow, device = 'cpu', dtype = torch.float32) -> torch.Tensor:
        '''
        Wrap from vol0 (reference volume) to vol1 (deformed volume) based on a dense flow field
        '''
        # Check if the flow is a PyTorch tensor
        if not torch.is_tensor(flow) or not torch.is_tensor(vol0):
            raise TypeError("The input flow must be a PyTorch tensor.")
        
        # vol0 dimension should be (D, H, W)
        shape = vol0.shape

        # Define the shape of Tensor
        shape_tensor = torch.tensor(shape, 
                                    dtype = dtype, 
                                    device = device, 
                                    requires_grad = False)

        # Permute the channel dimension to the end of flow field
        # Broadcast to all of t he dimensions
        # flow dimension should be (D, H, W)
        flow = torch.permute(flow, dims = (1, 2, 3, 0)) / (shape_tensor[None, None, None, :] / 2)
        flow += VolumeWarping.get_flow_grid_new(shape, norm_range=(-1, 1)).to(device, 
                                                                              dtype = dtype, 
                                                                              non_blocking=False).requires_grad_(False)

        # Warp by PyTorch function
        # Add (batch, channels) to the front of vol0
        # Add (batch) to the front of flow
        vol1 = F.grid_sample(input = torch.permute(vol0, dims = (2, 1, 0)).unsqueeze(0).unsqueeze(0),
                            grid = torch.permute(flow, dims = (2, 1, 0, 3)).unsqueeze(0),
                            mode = 'bilinear',
                            padding_mode = 'zeros',
                            align_corners = True)
        
        # Do a permute for output of grid_sample
        return torch.permute(vol1, dims = (0, 1, 4, 3, 2))

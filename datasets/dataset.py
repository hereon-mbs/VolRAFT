import torch
import numpy as np

from abc import ABC, abstractmethod
from torch.utils.data import Dataset

from utils.volume import DVCVolume
from utils.mask import DVCMask
from utils.flow import DVCFlow

from utils.file_handler import FileHandler
from utils.yaml_handler import YAMLHandler

# class DatasetFactory:
#     """
#     Factory to create dataset instances based on type.
#     """
#     @staticmethod
#     def get_dataset(dataset_type, *args, **kwargs):
#         if dataset_type == "volume":
#             return DVCVolumeDataset(*args, **kwargs)
#         elif dataset_type == "patch":
#             return DVCPatchDataset(*args, **kwargs)
#         else:
#             raise ValueError("Unknown dataset type")

class DVCDataset(Dataset, ABC):
    """
    A base class for all DVC dataset types in a PyTorch context, such as volumetric data or patch-based data.
    Subclasses must implement the __len__ and __getitem__ methods according to PyTorch's Dataset requirements.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __len__(self):
        """
        Should return the number of items in the dataset.
        """
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """
        Should retrieve an item by its index.
        """
        pass

    # Additional common methods for DVC datasets can be defined here.

class DVCVolumeDataset(DVCDataset):
    """
    Concrete implementation of DVCDataset for handling volumetric data.
    """
    def __init__(self, 
                 vol0_path : str, 
                 vol1_path : str, 
                 mask_path : str, 
                 flow_path : str = None, 
                 transforms = None):
        super().__init__()
        
        self.vol0 = DVCVolume()
        self.vol0.load_data(folder_path = vol0_path)

        self.vol1 = DVCVolume()
        self.vol1.load_data(folder_path = vol1_path)

        self.mask = DVCMask()
        self.mask.load_data(folder_path = mask_path)

        
        if flow_path is not None:
            self.flow = DVCFlow()
            self.flow.load_data(folder_path = flow_path)
        else:
            self.flow = None

        self.transforms = transforms

    def __len__(self):
        # For now: use one
        # should return more later on
        return 1

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transforms:
            sample = self.transforms(sample)
        return sample

class DVCPatchDataset(DVCDataset):
    """
    Handles patch-based data for volumetric images, masks, and flows, loading from .npy files as specified in a YAML configuration.
    """
    def __init__(self, dataset_path, patch_shape=None):
        """
        Initializes the dataset with a YAML configuration file and optional patch shape.
        
        Parameters:
            dataset_path (str): Path to the YAML configuration file specifying dataset directories.
            patch_shape (tuple of int, optional): Desired patch shape (nx, ny, nz). Uses the full patch shape from the first patch if None.
        """
        super().__init__()
        self.dataset_path = dataset_path

        self.patch_shape = np.array(patch_shape, dtype=int) if patch_shape is not None else None

        self.__patch_shape_full_4d = None  # Private: Full patch shape including channel dimension
        self.__patch_shape_full_3d = None  # Private: Full patch shape excluding channel dimension

        self.reset(patch_shape = self.patch_shape)

    def __len__(self):
        """
        Returns the total number of patches in the dataset.
        """
        return self.num_patches
    
    def __getitem__(self, idx):
        """
        Retrieves a dataset item by index.
        
        Parameters:
            idx (int): Index of the item to retrieve.
            
        Returns:
            tuple: Tensors for vol0, vol1, mask, and flow patches at the specified index.
        """
        # Define a local function to simplify repetitive loading and slicing operations.
        def load_and_slice(path):
            data = np.load(path)
            if self.roi_start is not None and self.roi_end is not None:
                data = data[:, self.roi_start[0]:self.roi_end[0], self.roi_start[1]:self.roi_end[1], self.roi_start[2]:self.roi_end[2]]
            return torch.from_numpy(data)

        vol0 = load_and_slice(self.paths_vol0[idx])
        vol1 = load_and_slice(self.paths_vol1[idx])
        mask = load_and_slice(self.paths_mask[idx])
        flow = load_and_slice(self.paths_flow[idx])

        return vol0, vol1, mask, flow
    
    @property
    def nc(self):
        """
        Retrieves the number of channels in the patch data.

        This property reflects the first dimension of the 4D patch shape, 
        indicating the number of channels in each patch. It is determined based
        on the full patch shape extracted from the first patch in the dataset.

        Returns:
            int: The number of channels in the patch data.
        """
        if self.__patch_shape_full_4d is None:
            raise ValueError("Patch shape is unknown. Ensure 'fetch' is called before.")
        
        return self.__patch_shape_full_4d[0]

    @property
    def nx(self):
        """
        Retrieves the patch size along the x-axis.

        This property reflects the size of the patch along the x-axis (width),
        which is part of the user-defined or automatically determined 3D patch shape.

        Returns:
            int: The size of the patch along the x-axis.
        """
        if self.patch_shape is None:
            raise ValueError("Patch shape is unknown. Ensure 'fetch' is called before.")
        
        return self.patch_shape[0]

    @property
    def ny(self):
        """
        Retrieves the patch size along the y-axis.

        This property reflects the size of the patch along the y-axis (height),
        which is part of the user-defined or automatically determined 3D patch shape.

        Returns:
            int: The size of the patch along the y-axis.
        """
        if self.patch_shape is None:
            raise ValueError("Patch shape is unknown. Ensure 'fetch' is called before.")
        
        return self.patch_shape[1]

    @property
    def nz(self):
        """
        Retrieves the patch size along the z-axis.

        This property reflects the size of the patch along the z-axis (depth),
        which is part of the user-defined or automatically determined 3D patch shape.

        Returns:
            int: The size of the patch along the z-axis.
        """
        if self.patch_shape is None:
            raise ValueError("Patch shape is unknown. Ensure 'fetch' is called before.")
        
        return self.patch_shape[2]

    def reset(self, patch_shape=None):
        """
        Resets dataset paths and counters to initial empty state. Prepares for new dataset configuration.
        """
        self.num_patches = 0
        self.paths_vol0, self.paths_vol1, self.paths_mask, self.paths_flow = [], [], [], []

        self.patch_shape = np.array(patch_shape) if patch_shape is not None else None
        self.roi_start, self.roi_end, self.roi_center = None, None, None

    def calculate_roi(self):
        """
        Calculates the Region of Interest (ROI) start, end, and center based on the determined patch shape.
        This method sets the roi_start, roi_end, and roi_center attributes to define the ROI for patch extraction.
        """
        if self.patch_shape is None or (self.patch_shape == self.__patch_shape_full_3d).all():
            # Use the full patch shape if no specific patch_shape is provided or it matches the full shape
            self.roi_start = np.zeros(3, dtype=int)
            self.roi_end = self.__patch_shape_full_3d
        else:
            # Calculate the ROI for the specified patch_shape
            center_full = self.__patch_shape_full_3d // 2
            half_patch_shape = self.patch_shape // 2

            self.roi_start = center_full - half_patch_shape
            self.roi_end = center_full + half_patch_shape + self.patch_shape % 2  # Adjust for odd dimensions.

        # Calculate the center of the ROI, useful for centering operations or adjustments
        self.roi_center = (self.roi_end + self.roi_start) // 2

    def fetch_patch_shape(self):
        """
        Determines patch size from the dataset configuration or the first patch file.
        Sets ROI start, end, and center based on the determined or provided patch size.
        """
        if not self.paths_vol0:
            raise ValueError("Volume paths are empty. Ensure 'fetch' is called before 'fetch_patch_shape'.")

        # Load the first volume to determine full patch shape.
        vol0_first = np.load(self.paths_vol0[0])

        if vol0_first.ndim == 3:  # Assuming the data doesn't include channel dimension
            vol0_first = np.expand_dims(vol0_first, axis=0)  # Add a channel dimension

        self.__patch_shape_full_4d = np.array(vol0_first.shape)
        self.__patch_shape_full_3d = self.__patch_shape_full_4d[1:]  # Exclude channel dimension

        if self.patch_shape is None:
            self.patch_shape = self.__patch_shape_full_3d
        else:
            if (self.patch_shape <= 0).any() or (self.patch_shape > self.__patch_shape_full_3d).any():
                raise ValueError(f"Invalid patch_shape {self.patch_shape}. Must be positive and <= {self.__patch_shape_full_3d}.")

        self.calculate_roi()

    def fetch(self) -> None:
        """
        Loads dataset configuration from a YAML file and populates paths for volumes, masks, and flows.
        """
        if not self.dataset_path or not FileHandler.exists(self.dataset_path) or not FileHandler.has_extension(self.dataset_path, ['.yaml', '.yml']):
            raise ValueError(f"Invalid dataset path or file format: {self.dataset_path}")

        content = YAMLHandler.read_yaml(self.dataset_path)
        self.reset(patch_shape = self.patch_shape)

        datasets_train = content.get("datasets_train", [])
        for dataset_train in datasets_train:
            if dataset_train.get("enable", False):
                self.paths_vol0.extend(FileHandler.get_npy_paths(dataset_train["vol0_path"], verbose=False))
                self.paths_vol1.extend(FileHandler.get_npy_paths(dataset_train["vol1_path"], verbose=False))
                self.paths_mask.extend(FileHandler.get_npy_paths(dataset_train["mask_path"], verbose=False))
                self.paths_flow.extend(FileHandler.get_npy_paths(dataset_train["flow_path"], verbose=False))

        self.num_patches = len(self.paths_mask)
        if not all(len(lst) == self.num_patches for lst in [self.paths_vol0, self.paths_vol1, self.paths_flow]):
            raise ValueError('Mismatch in the number of patches among vol0, vol1, mask, and flow paths.')

        self.fetch_patch_shape()
    
if __name__ == "__main__":
    print("--- Example usage ---")

    dataset = DVCPatchDataset(dataset_path = "./datasets/datasets.yaml",
                              patch_shape=None)
    
    dataset.fetch()

    print(f'length = {len(dataset)}')
    print(f'patch shape = {dataset.patch_shape}')
    print(f'roi_start = {dataset.roi_start}')
    print(f'roi_end = {dataset.roi_end}')
    print(f'roi_center = {dataset.roi_center}')
    print(f'(C, X, Y, Z) = {dataset.nc}, {dataset.nx}, {dataset.ny}, {dataset.nz}')

    patch_vol0, patch_vol1, patch_mask, patch_flow = dataset[1000]
    print(f'example patch_vol0 shape = {patch_vol0.shape}')
    print(f'example patch_vol1 shape = {patch_vol1.shape}')
    print(f'example patch_mask shape = {patch_mask.shape}')
    print(f'example patch_flow shape = {patch_flow.shape}')

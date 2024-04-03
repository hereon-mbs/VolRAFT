import numpy as np

from scipy.ndimage import map_coordinates
from PIL import Image, TiffImagePlugin

from tqdm import tqdm

from abc import ABC, abstractmethod

from utils.file_handler import FileHandler

class Volume(ABC):
    """
    Abstract base class for volume data handling. Defines a common interface
    for volume transformations and operations.
    """
    
    @abstractmethod
    def rotate(self):
        pass

    @abstractmethod
    def load_data(self):
        pass

class DVCVolume(Volume):
    """
    Represents a volumetric dataset, typically used for handling 3D/4D other volumetric data.
    
    This class provides functionalities for loading, processing, and manipulating volumetric data stored in various formats, 
    including TIFF and NPY files. It supports operations such as fetching data from files, applying transformations, and 
    managing the data's underlying representation and dtype.

    Attributes:
        data_np (np.array): The numpy array representing the volume data. Its shape and dtype are determined by the input data and the npdtype parameter.
        indexing (str): The indexing format of the volume data, either 'dhw' (depth, height, width) for volumetric data or 'xyz' (width, height, depth) for spatial data.
        npdtype (data-type): The numpy data type of the volume data, allowing control over the precision and memory usage of the data representation.

    Parameters:
        volume_data (np.array, optional): A 3D numpy array representing the initial volume data. If not provided, the volume data is initialized as None.
        indexing (str, optional): Specifies the indexing format of the volume data ('dhw' or 'xyz'). Defaults to 'dhw'.
        npdtype (data-type, optional): Specifies the numpy data type for the volume data. Defaults to np.float32, suitable for most medical imaging tasks.
    
    Methods:
        fetch_tiff_files(tiff_paths): Loads volume data from a sequence of TIFF files specified by their paths.
        fetch_npy_files(npy_paths, idx=0): Loads volume data from a specified NPY file or the first file in a list of NPY file paths.
        load_data(folder_path, tiff_subfolder="slices_xyz", verbose=True): High-level method to load volume data from a directory, preferring NPY files but falling back to TIFF files if necessary.
    """

    def __init__(self, volume_data=None, indexing='dhw', npdtype=np.float32):
        """
        Initializes the DVCVolume instance with optional volume data, indexing format, and numpy data type.

        Args:
            volume_data (np.array, optional): 3D/4D numpy array representing the volume data. The data can be
                of any numpy-supported data type, but will be stored internally based on the `npdtype` argument.
                Defaults to None, in which case no volume data is initially set.
            indexing (str, optional): Specifies the indexing format of the volume data. It can be either 'dhw'
                (depth, height, width) or 'xyz' (width, height, depth), indicating the organization of the
                volume data dimensions. Defaults to 'dhw'.
            npdtype (data-type, optional): The desired data-type for the volume data stored within the class
                instance. The volume data will be converted to this numpy data type upon assignment. This allows
                controlling the precision and memory footprint of the volume data. Defaults to np.float32.

        Raises:
            ValueError: If the provided `indexing` is not one of the supported formats ('dhw' or 'xyz').
        """
        if indexing not in ['dhw', 'xyz']:
            raise ValueError("Indexing must be either 'dhw' or 'xyz'.")

        self.data_np = volume_data.astype(npdtype, copy = True) if volume_data is not None else None
        self.indexing = None if volume_data is None else indexing
        self.__npdtype = npdtype

    @property
    def data(self):
        """
        Alias for accessing `data_np` as `data`.
        """
        return self.data_np

    @data.setter
    def data(self, value):
        """
        Allows setting `data_np` through the `data` alias.
        """
        self.data_np = value
        self.indexing = None if value is None else self.indexing

    @property
    def shape(self):
        """
        Returns the shape of the `data_np` volume data.
        """
        if self.data_np is None:
            return None  # Or raise an error if preferred
        return self.data_np.shape
    
    @property
    def npdtype(self) -> np.dtype:
        """
        Returns the data type of the `data_np` volume data.
        """
        if self.data_np is None:
            return None
        else:
            return self.data_np.dtype
    
    @npdtype.setter
    def npdtype(self, value):
        """
        Allows setting `npdtype`.
        """
        if self.data_np is not None:
            self.__npdtype = value
            self.data_np = self.data_np.astype(self.__npdtype, copy = False)
    
    @property
    def dtype(self) -> np.dtype:
        """
        Alias for accessing `npdtype` as `dtype`.
        """
        return self.npdtype
    
    @dtype.setter
    def dtype(self, value):
        """
        Alias for accessing `npdtype` as `dtype`.
        """
        self.npdtype(value)

    @staticmethod
    def add_channel_dims(array):
        return np.expand_dims(array, 0)

    def to_dhw(self) -> None:
        """
        Transposes the volume data to (d, h, w) format.
        """
        if self.data_np is None:
            raise ValueError("Volume data is not initialized.")
        
        if len(self.data_np.shape) != 3 and len(self.data_np.shape) != 4:
            raise ValueError("Volume data must be a 3D or 4D array.")
        
        if self.indexing == 'xyz' or self.indexing == None:
            if len(self.data_np.shape) == 3:
                self.data_np = np.transpose(self.data_np, axes=(2, 1, 0))
            elif len(self.data_np.shape) == 4:
                self.data_np = np.transpose(self.data_np, axes=(0, 3, 2, 1))
            else:
                raise ValueError("Volume data must be a 3D or 4D array.")
        
        self.indexing = 'dhw'

        return None
    
    def to_xyz(self) -> None:
        """
        Transposes the volume data from to (x, y, z) format.
        """
        if self.data_np is None:
            raise ValueError("Volume data is not initialized.")
        
        if len(self.data_np.shape) != 3 and len(self.data_np.shape) != 4:
            raise ValueError("Volume data must be a 3D or 4D array.")
        
        if self.indexing == 'dhw' or self.indexing == None:
            if len(self.data_np.shape) == 3:
                self.data_np = np.transpose(self.data_np, axes=(2, 1, 0))
            elif len(self.data_np.shape) == 4:
                self.data_np = np.transpose(self.data_np, axes=(0, 3, 2, 1))
            else:
                raise ValueError("Volume data must be a 3D or 4D array.")
        
        self.indexing = 'xyz'

        return None

    def rotate(self, angle, unit='radians'):
        """
        Rotates the volume data by specified angles around the x, y, and z axes.

        This method applies a 3D rotation to the volume data using a sequence of
        rotation angles around the x/d, y/h, and z/w axes (in that order). The rotation
        is performed around the center of the volume data. Angles can be specified
        in either radians or degrees.

        Parameters:
        angle (tuple or np.ndarray): Rotation angles around the x/d, y/h, and z/w axes.
                                     Can be a tuple or a numpy array of shape (3,).
        unit (str, optional): The unit of the angles. Can be 'radians' or 'degrees'.
                              Default is 'radians'.

        Returns:
        np.ndarray: The rotated volume data.

        Note:
        The function automatically converts angles from degrees to radians if 'unit'
        is set to 'degrees'.
        """
        if self.data_np is None:
            raise ValueError("Volume data is not initialized.")
        
        if len(self.data_np.shape) != 3 and len(self.data_np.shape) != 4:
            raise ValueError("Volume data must be a 3D or 4D array.")
        
        if not isinstance(angle, np.ndarray):
            angle = np.array(angle)

        if angle.shape != (3,):
            raise ValueError(
                "Angle must be a tuple or numpy array of shape (3,)"
                " representing rotation angles for the x, y, and z axes."
            )

        if unit == 'degrees':
            angle = np.radians(angle)
        elif unit != 'radians':
            raise ValueError("Unit must be either 'radians' or 'degrees'.")

        R0 = np.array([
            [1, 0, 0],
            [0, np.cos(angle[0]), -np.sin(angle[0])],
            [0, np.sin(angle[0]),  np.cos(angle[0])]
        ])

        R1 = np.array([
            [np.cos(angle[1]), 0, np.sin(angle[1])],
            [0, 1, 0],
            [-np.sin(angle[1]), 0, np.cos(angle[1])]
        ])

        R2 = np.array([
            [np.cos(angle[2]), -np.sin(angle[2]), 0],
            [np.sin(angle[2]),  np.cos(angle[2]), 0],
            [0, 0, 1]
        ])

        # Squeeze the first dimension if needed
        if len(self.data_np.shape) == 4:
            if self.data_np.shape[0] == 1:
                data_np = np.squeeze(self.data_np, axis=0)
            else:
                raise ValueError(
                    f"Volume data has the shape {self.data_np.shape}"
                    ", but the 4D array must have shape=1 at the first dimension."
                )
        else:
            data_np = self.data_np

        R = R2 @ R1 @ R0

        N0, N1, N2 = data_np.shape
        n0, n1, n2 = np.mgrid[0:N0, 0:N1, 0:N2]
        indices = np.reshape(np.stack([n0, n1, n2]), (3, -1))

        center = np.array([N0 / 2.0, N1 / 2.0, N2 / 2.0]).reshape(3, 1)
        indices_centered = indices - center

        rotated_indices = R @ indices_centered + center
        rotated_indices_reshaped = np.reshape(rotated_indices, (3, N0, N1, N2))

        rotated_volume = map_coordinates(
            data_np, rotated_indices_reshaped, order=1, mode='constant'
        )

        return rotated_volume
    
    def get_memory_usage_gb(self, verbose=True):
        """
        Calculates and optionally prints the memory usage of the volume data.

        This method calculates the memory usage of the `data_np` numpy array
        in gigabytes (GB). If `verbose` is True, the memory usage is printed.

        Parameters:
        verbose (bool, optional): If True, prints the memory usage. Default is True.

        Returns:
        float: The memory usage of the `data_np` array in GB.
        """
        if self.data_np is None:
            return 0
        
        # Calculating the memory usage
        memory_usage = self.data_np.nbytes  # Memory usage in bytes
        memory_usage_gb = memory_usage / (1024.0 ** 3)  # Convert bytes to GB

        if verbose:
            print(f"Memory usage of the array: {memory_usage} bytes "
                  f"({memory_usage_gb:.2f} GB)")

        return memory_usage_gb
    
    def fetch_tiff_files(self, tiff_paths):
        """
        Fetches TIFF files into a numpy array.
        
        Parameters:
            tiff_paths (list of str): Paths to TIFF files to be loaded.
        
        Returns:
            tuple: A numpy array of loaded images, indexing format, and status message.
        """
        if not tiff_paths:
            return None, 'xyz', "No TIFF paths provided."

        # Initialize variables for dimensions
        data_np = None

        # Load files into a numpy array
        for idx, tiff_file in enumerate(tqdm(tiff_paths, desc="Loading TIFF files")):
            file_image = np.array(Image.open(tiff_file))

            # Initialize array on first iteration
            if data_np is None:
                data_np = np.zeros(file_image.shape + (len(tiff_paths),), dtype=self.npdtype)

            data_np[..., idx] = file_image

        folder_path = FileHandler.extract_folder_path(tiff_paths[0])
        message = f"Loaded XYZ data from {len(tiff_paths)} TIFF files at {folder_path}"
        return data_np, 'xyz', message
    
    def fetch_npy_files(self, npy_paths, idx=0):
        """
        Fetches a numpy file and loads it into the class instance.
        
        Parameters:
            npy_paths (list of str): Paths to .npy files to be loaded.
            idx (int, optional): Index of the .npy file in the list to load.
        
        Returns:
            tuple: A numpy array of loaded data, indexing format, and status message.
        """
        if not npy_paths:
            return None, 'dhw', "No NPY paths provided."

        data_np = np.load(npy_paths[idx])
        message = f"Loaded DHW data from {npy_paths[idx]}"
        return data_np, 'dhw', message
    
    def load_data(self, folder_path, tiff_subfolder="slices_xyz", verbose=True) -> None:
        """
        Loads data from specified folder path, preferring .npy files, falling back to TIFF.
        
        Parameters:
            folder_path (str): Path to the folder containing data files.
            tiff_subfolder (str, optional): Subfolder name for TIFF files.
            verbose (bool, optional): If True, prints loading progress and messages.
        """
        # Assume FileHandler methods are correctly defined elsewhere
        npy_paths = FileHandler.get_npy_paths(folder_path, verbose=False)

        if npy_paths:
            self.data_np, self.indexing, message = self.fetch_npy_files(npy_paths)
        else:
            subfolder_path = FileHandler.find_subfolder(folder_path, pattern=tiff_subfolder, verbose=False)
            if subfolder_path:
                tiff_paths = FileHandler.get_tiff_paths(subfolder_path, verbose=False)
                self.data_np, self.indexing, message = self.fetch_tiff_files(tiff_paths)
            else:
                raise FileNotFoundError(f"Subfolder {tiff_subfolder} does not exist in {folder_path}")

        self.data_np = self.data_np.astype(self.__npdtype, copy = False)

        if verbose and message:
            print(message)

    # Function to save volume to slices
    # Save to folder
    def save_volume_to_slices(self, target_path, prefix, num_zero = 4, suffix = 'tif', verbose = True):
        # Store the original indexing
        original_indexing = self.indexing

        # Convert self to xyz
        self.to_xyz()

        # Get the data shape
        nx, ny, nz = self.data_np.shape

        # Define the axis of slices (which is z-axis here)
        axis_slice = 2

        # Create an IFD object for TIFF tags
        ifd = TiffImagePlugin.ImageFileDirectory_v2()

        # TIFFTAG_BITSPERSAMPLE: 8 bits per sample
        ifd[TiffImagePlugin.BITSPERSAMPLE] = 32
        ifd[TiffImagePlugin.SAMPLESPERPIXEL] = 1
        ifd[TiffImagePlugin.IMAGELENGTH] = nx
        ifd[TiffImagePlugin.IMAGEWIDTH] = ny

        # for each slice at axis z
        for idx in np.arange(self.data_np.shape[axis_slice]):
            # Get the numpy array of slice
            img_array = self.data_np[:, :, idx].squeeze()
            
            # Convert to image object
            img = Image.fromarray(img_array)

            # Get the full path of file
            filepath = FileHandler.join(target_path, f'{prefix}{idx:0>{num_zero}}.{suffix}')

            # Get the folder path
            # folder = os.path.dirname(os.path.abspath(filepath))
            folder = FileHandler.extract_folder_path(FileHandler.realpath(filepath))

            # Prepare the folder if not existed
            if not FileHandler.exists(folder):
                FileHandler.mkdir(folder)

            # Convert to floating point
            img = img.convert("F")

            # Save to the filepath
            # img.save(fp = filepath)
            img.save(fp = filepath, format = 'TIFF', tiffinfo = ifd)

            # Show the result
            if verbose:
                print(f'saved to {filepath}')

        # Convert back to original indexing
        if original_indexing == 'dhw':
            self.to_dhw()

        return
    
if __name__ == "__main__":
    print("--- Example usage ---")

    # path = "/asap3/petra3/gpfs/p05/2021/data/11008741/processed/wongtakm/datasets/ki4d4e/dvc/volume_960x1280x1280/2021_11008741_syn0154_103L_Mg5Gd_4w_000_fs004/volume0/vol0_960x1280x1280.npy"
    # path = "/asap3/petra3/gpfs/p05/2021/data/11008741/processed/wongtakm/datasets/ki4d4e/dvc/volume_960x1280x1280/2021_11008741_syn0154_103L_Mg5Gd_4w_000_fs004/ground_truth/flow/flow_960x1280x1280.npy"
    # volume_data = np.load(path)
    
    folder_path = "/asap3/petra3/gpfs/p05/2021/data/11008741/processed/wongtakm/datasets/ki4d4e/dvc/volume_960x1280x1280/2021_11008741_syn0154_103L_Mg5Gd_4w_000_fs004/volume0"
    # folder_path = "/asap3/petra3/gpfs/p05/2021/data/11008741/processed/wongtakm/datasets/ki4d4e/dvc/volume_960x1280x1280/2021_11008741_syn0154_103L_Mg5Gd_4w_000_fs004/mbsoptflow"

    vol0 = DVCVolume()
    vol0.load_data(folder_path, tiff_subfolder="dx")

    print(vol0.shape)
    print(vol0.indexing)

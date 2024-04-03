import numpy as np

from PIL import Image, TiffImagePlugin

from utils.file_handler import FileHandler

from utils.volume import DVCVolume

class DVCFlow(DVCVolume):
    """
    Specialized DVCVolume class for handling flow data.
    Inherits from DVCVolume and adds or overrides methods specific to mask operations.
    """
    def __init__(self, flow_data=None, indexing='dhw', npdtype=np.float32):
        """
        Initializes the DVCFlow instance with optional mask data.
        
        Parameters:
            flow_data (np.array, optional): 4D array representing the mask data.
            indexing (str, optional): Specifies the indexing format ('dhw' or 'xyz').
            npdtype (data-type, optional): Numpy data type for mask data storage, default is np.float32.
        """
        # Initialize the superclass with the provided parameters
        super().__init__(flow_data, indexing, npdtype)

    def load_data(self, folder_path, verbose=True) -> None:
        """
        Loads flow data from a specified folder path with a specific order of preference:
        1. .npy files directly within the folder.
        2. Gradient folders ('dx', 'dy', 'dz') within a 'slices_xyz' subfolder.
        3. Gradient folders ('dx', 'dy', 'dz') directly within the folder.

        Parameters:
            folder_path (str): Path to the folder containing data files.
            verbose (bool, optional): If True, prints loading progress and messages. Defaults to True.
        """
        npy_paths = FileHandler.get_npy_paths(folder_path, verbose=verbose)

        if npy_paths:
            self.data_np, self.indexing, message = self.fetch_npy_files(npy_paths)
            if verbose:
                print(message)
            return

        # Attempt to load from 'slices_xyz' subfolder first
        slices_xyz_path = FileHandler.find_subfolder(folder_path, pattern="slices_xyz", recursive=False, verbose=verbose)
        grad_folders = ["dx", "dy", "dz"]
        gradients = []

        # Define a helper function to attempt loading gradients
        def attempt_load_gradients(base_path):
            loaded_gradients = []
            for grad_folder in grad_folders:
                grad_folder_path = FileHandler.find_subfolder(base_path, pattern=grad_folder, recursive=False, verbose=verbose)
                if grad_folder_path:
                    tiff_paths = FileHandler.get_tiff_paths(grad_folder_path, verbose=verbose)
                    if tiff_paths:
                        grad_data, _, message = self.fetch_tiff_files(tiff_paths)
                        loaded_gradients.append(grad_data)
                        if verbose:
                            print(f"Loaded {grad_folder.upper()} data: {message}")
                    else:
                        if verbose:
                            print(f"No TIFF files found in {grad_folder_path}")
            return loaded_gradients

        # Try loading from 'slices_xyz' if exists
        if slices_xyz_path:
            gradients = attempt_load_gradients(slices_xyz_path)
        
        # If not all gradients were loaded and 'slices_xyz' was checked, attempt from the root folder
        if len(gradients) < 3:
            gradients = attempt_load_gradients(folder_path)
        
        # Final check if gradients are loaded
        if len(gradients) == 3:
            self.data_np = np.stack(gradients, axis=0)
            self.indexing = 'xyz'
        else:
            raise FileNotFoundError("Could not find all gradient data (dx, dy, dz) in the specified locations.")
        
    # Save to folder
    def save_volume_to_slices(self, target_path, prefix, num_zero = 4, suffix = 'tif', verbose = True):
        raise NotImplementedError('save_volume_to_slices is not implemented in DVCFlow. Use save_flow_to_slices instead')
        
    # Save to folder
    def save_flow_to_slices(self, target_path, prefix = ("ux_", "uy_", "uz_"), num_zero = 4, suffix = 'tif', subfolder = ("dx", "dy", "dz"), verbose = True):
        # Store the original indexing
        original_indexing = self.indexing

        # Convert self to xyz
        self.to_xyz()

        # Get the data shape
        nc, nx, ny, nz = self.data_np.shape

        # Define the axis of slices (which is z-axis here)
        axis_slice = 3

        # Create an IFD object for TIFF tags
        ifd = TiffImagePlugin.ImageFileDirectory_v2()

        # TIFFTAG_BITSPERSAMPLE: 8 bits per sample
        ifd[TiffImagePlugin.BITSPERSAMPLE] = 32
        ifd[TiffImagePlugin.SAMPLESPERPIXEL] = 1
        ifd[TiffImagePlugin.IMAGELENGTH] = nx
        ifd[TiffImagePlugin.IMAGEWIDTH] = ny

        # Special operation to synchronize data format for voxel2mesh
        flow_data = np.zeros_like(self.data) # TODO: Find out why
        flow_data[0, :, :, :] =  1.0 * self.data[1, :, :, :] # TODO: Find out why
        flow_data[1, :, :, :] = -1.0 * self.data[2, :, :, :] # TODO: Find out why
        flow_data[2, :, :, :] = -1.0 * self.data[0, :, :, :] # TODO: Find out why

        # for each direction
        for axis in np.arange(3):
            # for each slice at axis z
            for idx in np.arange(flow_data.shape[axis_slice]):
                # Get the numpy array of slice
                img_array = flow_data[axis, :, :, idx].squeeze()
                
                # Convert to image object
                img = Image.fromarray(img_array)

                # Get the full path of file
                filepath = FileHandler.join(target_path, f'{subfolder[axis]}', f'{prefix[axis]}{idx:0>{num_zero}}.{suffix}')

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
    
    folder_path = "/asap3/petra3/gpfs/p05/2021/data/11008741/processed/wongtakm/datasets/ki4d4e/dvc/volume_960x1280x1280/2021_11008741_syn0154_103L_Mg5Gd_4w_000_fs004/ground_truth/flow"
    # folder_path = "/asap3/petra3/gpfs/p05/2021/data/11008741/processed/wongtakm/datasets/ki4d4e/dvc/volume_960x1280x1280/2021_11008741_syn0154_103L_Mg5Gd_4w_000_fs004/mbsoptflow"

    flow = DVCFlow()
    flow.load_data(folder_path)

    print(flow.shape)
    print(flow.indexing)

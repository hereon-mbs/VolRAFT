import numpy as np

from scipy.ndimage import binary_closing, generate_binary_structure

from utils.volume import DVCVolume

class DVCMask(DVCVolume):
    """
    Specialized DVCVolume class for handling mask data.
    Inherits from DVCVolume and adds or overrides methods specific to mask operations.
    """
    def __init__(self, mask_data=None, indexing='dhw', npdtype=np.float32):
        """
        Initializes the DVCMask instance with optional mask data.
        
        Parameters:
            mask_data (np.array, optional): 3D/4D array representing the mask data, typically binary.
            indexing (str, optional): Specifies the indexing format ('dhw' or 'xyz').
            npdtype (data-type, optional): Numpy data type for mask data storage, default is np.float32.
        """
        # Initialize the superclass with the provided parameters
        super().__init__(mask_data, indexing, npdtype)

    @staticmethod
    def get_foreground_mask(mask, labels, closing_percent=1.5, npdtype=np.float32):
        """
        Generates a foreground mask by applying morphological closing and filtering by labels.

        Parameters:
            mask (DVCMask): An instance of DVCMask containing the original mask data.
            labels (list or array): A list or array of labels to filter the foreground.
            closing_percent (float, optional): The percentage to calculate the half-width for closing. Defaults to 1.5.
            npdtype (data-type, optional): The numpy data type for the output mask. Defaults to np.float32.

        Returns:
            DVCMask: A new DVCMask instance containing the processed foreground mask.
        """
        return DVCMask(data_np=mask.get_foreground(labels, closing_percent, npdtype),
                       indexing=mask.indexing,
                       npdtype=npdtype)

    def closing(self, mask_binary, percent=1.5, verbose=False):
        """
        Applies morphological closing to the binary mask.

        Parameters:
            mask_binary (np.array): The binary mask to which closing is applied.
            percent (float, optional): The percentage used to calculate the closing's half-width. Defaults to 1.5.
            verbose (bool, optional): If True, prints additional information. Defaults to False.

        Returns:
            np.array: The binary mask after applying closing.
        """
        should_unsqueeze = len(mask_binary.shape) == 4
        if should_unsqueeze:
            mask_binary = np.squeeze(mask_binary, axis=0)

        mask_dim = np.sqrt(np.mean(np.array(mask_binary.shape) ** 2))
        half_width = np.maximum(1, np.round(mask_dim * percent / 100)).astype(int)

        if verbose:
            print(f'Applying closing with half-width = {half_width}')

        structure = generate_binary_structure(rank=3, connectivity=3)
        mask_binary_closed = binary_closing(mask_binary.astype(bool), structure=structure, iterations=half_width)

        return np.expand_dims(mask_binary_closed, axis=0) if should_unsqueeze else mask_binary_closed


    def get_foreground(self, labels, closing_percent=1.5, npdtype=np.bool_):
        """
        Generates a binary foreground mask based on specified labels.

        Parameters:
            labels (list or array): Labels to include in the foreground.
            closing_percent (float, optional): Percentage for the morphological closing operation. Defaults to 1.5.
            npdtype (data-type, optional): Data type for the output mask. Defaults to np.bool_.

        Returns:
            np.array: The foreground mask as a binary array.
        """
        mask_fg_binary = np.zeros_like(self.data_np, dtype=bool)

        for value in labels:
            mask_fg_binary |= self.data_np == value

        if closing_percent > 0.0:
            mask_fg_binary = self.closing(mask_binary=mask_fg_binary, percent=closing_percent, verbose=False)

        return mask_fg_binary.astype(npdtype)
    
    def load_data_labels(self, folder_path, labels, closing_percent=1.5, tiff_subfolder="slices_xyz", npdtype=np.bool_, verbose=True):
        """
        Loads data from a specified folder and applies label-based foreground extraction and closing.

        Parameters:
            folder_path (str): Path to the folder containing data files.
            labels (list or array): Labels to filter for the foreground.
            closing_percent (float, optional): The percentage for closing calculation. Defaults to 1.5.
            npdtype (data-type, optional): Numpy data type for the foreground mask. Defaults to np.bool_.
            verbose (bool, optional): If True, prints additional information. Defaults to True.
        """
        self.load_data(folder_path=folder_path, tiff_subfolder=tiff_subfolder, verbose=verbose)

        self.data_np = self.get_foreground(labels=labels, closing_percent=closing_percent, npdtype=npdtype)

        if verbose:
            print(f'Foreground extracted for labels {labels} with closing percent = {closing_percent}')
    
if __name__ == "__main__":
    print("--- Example usage ---")
    
    folder_path = "/asap3/petra3/gpfs/p05/2021/data/11008741/processed/wongtakm/datasets/ki4d4e/dvc/volume_960x1280x1280/2021_11008741_syn0154_103L_Mg5Gd_4w_000_fs004/mask"

    mask = DVCMask()
    mask.load_data_labels(folder_path, labels = [1, 2], closing_percent = 0.1, verbose = True)

    print(mask.shape)
    print(mask.indexing)

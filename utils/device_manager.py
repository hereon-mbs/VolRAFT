import os
import gc
import torch

class DeviceManager:
    """
    Manages computing devices for operations, allowing dynamic selection of CPU or CUDA devices based on availability and user specification.
    """

    # Get all garbage collection
    @staticmethod
    def garbage() -> None:
        # Python garbage collection
        gc.collect()

        # PyTorch garbage collection
        torch.cuda.empty_cache()

        return

    @staticmethod
    def get_device_string(device) -> str:
        """
        Extracts the string representation of a PyTorch device from various input types.
        
        Parameters:
        - device (torch.device, torch.cuda.Device, str): The input device representation.
        
        Returns:
        - str: The string representation of the device.
        
        Raises:
        - ValueError: If the input device is not a recognized type.
        """
        if isinstance(device, torch.device):
            # Input is an instance of torch.device
            return str(device)
        elif isinstance(device, torch.cuda.Device):
            # Input is an instance of torch.cuda.Device
            return device.type + ":" + str(device.index)
        elif isinstance(device, str):
            # Input is already a string
            return device
        else:
            # Input is not a recognized type
            raise ValueError("The input device must be a torch.device, torch.cuda.Device, or a string.")

    def __init__(self, device_number=None, verbose=True):
        """
        Initializes the DeviceManager with an optional specific CUDA device number and verbosity control.
        
        Parameters:
            device_number (int, optional): The number of the CUDA device to use if available. If None, the device with the most free memory is selected.
            verbose (bool, optional): Controls the verbosity of the print statements. Defaults to True.
        """
        self.device_number = device_number
        self.verbose = verbose
        self.device = self._set_device()

    def get_num_workers(self):
        """
        Determines the optimal number of workers for data loaders based on the available CPU cores.
        
        Returns:
            int: The recommended number of workers.
        """
        cpu_count = os.cpu_count() or 1  # Fallback to 1 if os.cpu_count() is None
        num_workers = max(cpu_count // 4, 1)  # Ensure at least one worker
        if self.verbose:
            print(f'Number of CPU cores = {cpu_count} => Recommended num_workers = {num_workers}')
        return num_workers

    def _set_device(self):
        """
        Sets the computing device based on availability and the optional device number.
        
        Returns:
            torch.device: The selected computing device.
        """
        if torch.cuda.is_available():
            if self.device_number is not None and self.device_number < torch.cuda.device_count():
                if self.verbose:
                    print(f"Using CUDA device: cuda:{self.device_number}")
                return torch.device(f'cuda:{self.device_number}')
            elif self.device_number is None:
                device_id, _ = self._get_device_with_max_memory()
                self.device_number = device_id
                if self.verbose:
                    print(f"Automatically selected CUDA device with maximum free memory: cuda:{device_id}")
                return torch.device(f'cuda:{device_id}')
            else:
                if self.verbose:
                    print(f"Specified CUDA device {self.device_number} is not available. Using CPU instead.")
        else:
            if self.verbose:
                print("CUDA is not available. Using CPU.")
        return torch.device('cpu')

    def _get_device_with_max_memory(self):
        """
        Identifies the CUDA device with the most free memory.
        
        Returns:
            tuple: The device ID with the most free memory, and the amount of free memory in GB.
        """
        max_memory = 0
        best_device = 0
        for device in range(torch.cuda.device_count()):
            torch.cuda.set_device(device)
            free_memory = torch.cuda.mem_get_info()[0]
            if free_memory > max_memory:
                max_memory = free_memory
                best_device = device
        return best_device, max_memory / (1024 ** 3)

    def get_device(self, use_cpu=False):
        """
        Retrieves the currently selected computing device.
        
        Returns:
            torch.device: The current computing device.

        Parameters:
            use_cpu (bool, optional): Controls whether to use CPU as the computing device. Defaults to False.
        """
        if use_cpu:
            return torch.device('cpu')
        else:
            return self.device
    
    def get_device_id(self):
        """
        Retrieves the ID of the currently selected CUDA device, if any.
        
        Returns:
            int or None: The CUDA device number, or None if CPU is used.
        """
        return self.device_number if self.device.type == 'cuda' else None
    
    def get_device_count(self):
        """
        Retrieves the total number of available CUDA devices.
        
        Returns:
            int: The number of available CUDA devices.
        """
        return torch.cuda.device_count()


if __name__ == "__main__":
    # # Example usage:
    # # Automatically select the best available device or manually specify a device number
    # # Replace 'DeviceController()' with 'DeviceController(device_number=your_device_number)' if specifying a device number
    print("--- Example usage ---")
    device_manager = DeviceManager()  # Automatically select the best available device
    device = device_manager.get_device()
    print("Current device:", device)

    print("--- Example usage ---")
    device_manager = DeviceManager(device_number=1, verbose=False)  # Manually specify device number (e.g., CUDA device 0)
    device = device_manager.get_device()
    print("Current device:", device)

    num_workers = device_manager.get_num_workers()
    print(f"number of workers: {num_workers}")

    cpu = device_manager.get_device(use_cpu=True)
    print(f'cpu: {cpu}')
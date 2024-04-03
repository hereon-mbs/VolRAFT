import random
import numpy as np
import torch
import os

class SeedManager:
    """
    Manages the random seed for Python's random module, NumPy, and PyTorch to ensure reproducibility across computations.
    Optionally generates and sets a random seed if None or 0 is provided as the seed value. Stores the current seed for reference.
    """

    def __init__(self, seed=None, verbose=True, deterministic=True):
        """
        Initializes the SeedManager with optional settings for the seed, verbosity, and deterministic behavior.

        Parameters:
            seed (int, optional): The seed value to set for random operations. Generates a random seed if None or 0.
            verbose (bool, optional): Controls the verbosity of the print statements. Defaults to True.
            deterministic (bool, optional): If True, attempts to make operations deterministic. Defaults to True.
        """
        self.verbose = verbose
        self.deterministic = deterministic
        self.current_seed = self.set_seed(seed)
        if self.deterministic:
            self.set_deterministic_behavior()

    def set_seed(self, seed=None):
        """
        Sets the random seed for Python's random module, NumPy, and PyTorch. Generates a random seed if None or 0 is provided.

        Parameters:
            seed (int, optional): The seed value to set for random operations. If None or 0, a random seed is generated.

        Returns:
            int: The seed that was set.
        """
        if seed is None or seed == 0:
            # Get 4 bytes of random data
            random_bytes = os.urandom(4)
            # Convert bytes to an integer
            seed = int.from_bytes(random_bytes, "big")
            if self.verbose:
                print(f"Generated random seed: {seed}")

        random.seed(seed)  # Python's built-in random module
        np.random.seed(seed)  # NumPy
        torch.manual_seed(seed)  # PyTorch

        # If using CUDA, also set the seed for all current and future GPUs
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        return seed

    def set_deterministic_behavior(self):
        """
        Enables deterministic behavior in PyTorch operations, if possible.
        Note: This may impact performance and is not guaranteed to work for all operations.
        """
        if not self.deterministic:
            return  # Return early if deterministic behavior is not desired

        # Set deterministic algorithms for convolution operations, if available
        if hasattr(torch.backends.cudnn, 'deterministic'):
            torch.backends.cudnn.deterministic = True

        if hasattr(torch.backends.cudnn, 'benchmark'):
            torch.backends.cudnn.benchmark = True
        
        if hasattr(torch, 'use_deterministic_algorithms'):
            # max_pool3d_with_indices_backward_cuda does not have a deterministic implementation
            # You can turn off determinism just for this operation, or you can use the 'warn_only=True' option, if that's acceptable for your application
            torch.use_deterministic_algorithms(False)

        if hasattr(torch.backends.cuda, 'matmul'):
            # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
            # in PyTorch 1.12 and later.
            torch.backends.cuda.matmul.allow_tf32 = True

        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        
        if self.verbose:
            print("Deterministic behavior is enabled for PyTorch operations.")

    def get_current_seed(self):
        """
        Retrieves the current seed value used for random operations.

        Returns:
            int: The current seed value.
        """
        return self.current_seed

if __name__ == "__main__":
    # Example usage:
    # Automatically generate a random seed or manually specify a seed number
    # Replace 'RandomSeedController()' with 'RandomSeedController(seed=your_seed_number)' if specifying a seed number
    print("--- Example usage ---")
    seed_manager = SeedManager()  # Automatically generate a random seed
    print(f"Current seed: {seed_manager.get_current_seed()}")

    print("--- Example usage ---")
    seed_manager = SeedManager(seed=42)  # Manually specify a seed number (e.g., seed=42)
    print(f"Current seed: {seed_manager.get_current_seed()}")
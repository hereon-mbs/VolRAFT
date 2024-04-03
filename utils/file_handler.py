import os
import shutil

class FileHandler:
    """Handles file and folder operations for the project."""

    @staticmethod
    def exists(path):
        """
        Checks whether a file or directory at the specified path exists.

        Parameters:
            path (str): The path to the file or directory to check.

        Returns:
            bool: True if the file or directory exists, False otherwise.
        """
        return os.path.exists(path)

    @staticmethod
    def has_extension(filepath, allowed_extensions):
        """
        Checks if the file at the given path has an allowed extension.

        Parameters:
            filepath (str): The path to the file.
            allowed_extensions (list or tuple): Allowed file extensions (include the dot, e.g., '.txt').

        Returns:
            bool: True if the file extension is allowed, False otherwise.
        """
        _, extension = os.path.splitext(filepath)
        return extension in allowed_extensions
    
    @staticmethod
    def mkdir(path, mode=0o711) -> None:
        try:
            os.makedirs(path, mode=mode, exist_ok=True)
        except FileExistsError:
            # If it's a directory, you might want to log it or handle it accordingly
            raise FileExistsError(f"Directory '{path}' already exists.")

        return
    
    @staticmethod
    def cp(src, dst) -> None:
        """
        Copies a file from the source path to the destination path.

        This method wraps `shutil.copyfile` to provide a straightforward interface for file copying
        within the FileHandler context. It ensures that the source file exists and the destination
        can be written to, logging appropriate messages in case of errors or success.

        Parameters:
            src (str): The path to the source file.
            dst (str): The path to the destination where the file should be copied.

        Raises:
            FileNotFoundError: If the source file does not exist.
            PermissionError: If there is a permission error accessing the source or writing to the destination.
            Exception: For other unforeseen errors related to file copying.

        Returns:
            None
        """

        if not os.path.exists(src):
            raise FileNotFoundError(f"Source file does not exist: {src}")

        if not os.path.isfile(src):
            raise ValueError(f"Source path is not a file: {src}")

        try:
            shutil.copyfile(src, dst)
        except PermissionError as e:
            raise PermissionError(f"Permission denied when copying from {src} to {dst}: {e}")
        except Exception as e:
            raise Exception(f"Error occurred when copying from {src} to {dst}: {e}")
        
    @staticmethod
    def ln(target, link_path) -> None:
        """
        Create symbolic link to the target path
        """
        if FileHandler.exists(target):
            return os.symlink(target, link_path)
        else:
            raise FileExistsError(f"{target} does not exist")
        
    @staticmethod
    def listdir(path) -> list[str]:
        return os.listdir(path)
    
    @staticmethod
    def mkdir(path) -> None:
        """
        Creates a directory at the specified path, including any necessary parent directories,
        with permissions allowing the owner full control and others to only execute/search.

        Parameters:
            path (str): The filesystem path where the directory (and any necessary parent directories)
                        should be created. The path can be absolute or relative.

        Raises:
            FileExistsError: If an attempt to create a directory fails because a file with the same
                             name already exists, or due to lack of permissions to create or access
                             part of the specified path.

        Note:
            - Directory permissions are set to 0o711.
            - This method is idempotent; it does not raise an exception if the directory already exists
              and matches the requested permissions.
        """
        try:
            os.makedirs(path, mode=0o711, exist_ok=True)
        except FileExistsError as e:
            # It's a rare case if 'exist_ok' is True; likely means 'path' is a file, not a directory.
            print(f"Failed to create directory '{path}': {e}")
            raise FileExistsError(f"Failed to create directory '{path}': it already exists as a file.")
        except PermissionError as e:
            # Handle lack of permissions
            print(f"Permission denied to create directory '{path}': {e}")
            raise PermissionError(f"Permission denied to create directory '{path}'.")
        except Exception as e:
            # Catch-all for other exceptions, such as a malformed path
            print(f"An unexpected error occurred while creating directory '{path}': {e}")
            raise Exception(f"An unexpected error occurred while creating directory '{path}': {e}")
        
    @staticmethod
    def join(*paths) -> str:
        """
        Concatenates multiple path components into a single path, ensuring the correct use of
        directory separators for the operating system. This method wraps `os.path.join` to provide
        a unified interface for path operations within the FileHandler.

        Parameters:
            *paths (str): An arbitrary number of string arguments representing individual path components.
                          These components will be joined in the order they are provided.

        Returns:
            str: A single string that represents the concatenated path, with appropriate directory
                 separators between components.

        Raises:
            ValueError: If any of the path components is None, raising an error to prevent unintended
                        path manipulation issues.

        Example:
            joined_path = FileHandler.join('/path/to', 'directory', 'filename.txt')
            # Output for UNIX-like systems: '/path/to/directory/filename.txt'
            # Output for Windows systems: '\\path\\to\\directory\\filename.txt'
        """
        # Validate input to ensure no component is None
        if any(component is None for component in paths):
            raise ValueError("NoneType found in path components. All components must be strings.")

        # Use os.path.join to concatenate the path components
        return os.path.join(*paths)

    @staticmethod
    def extract_folder_path(full_path):
        """
        Extracts the folder path from a full path to a file.

        Args:
            full_path (str): The full path to the file.

        Returns:
            str: The folder path extracted from the full path.
        """
        return os.path.dirname(full_path)
    
    @staticmethod
    def realpath(path):
        """
        Resolves and returns the canonical path of the specified filename, 
        resolving any symbolic links encountered in the path.

        This method uses `os.path.realpath` to follow any symbolic links to 
        determine the actual file or directory path that the symbolic link points to. 
        It resolves symbolic links, ../, and ./ segments to return the absolute path.

        Parameters:
        - path (str): The path to resolve. This can include symbolic links 
        or relative path segments like ../ or ./

        Returns:
        - str: The canonical path of the specified filename, with all symbolic links resolved.

        Example usage:
        real_path = FileHandler.realpath('/path/to/symbolic_link_or_directory')
        """
        return os.path.realpath(path)

    @staticmethod
    def get_paths(folder_path, extension, verbose=True):
        """Finds all file paths within a specified folder."""
        if not isinstance(folder_path, str):
            raise ValueError("Folder path must be a string.")
        
        if not os.path.isdir(FileHandler.realpath(folder_path)):
            raise ValueError(f"The specified folder does not exist: {folder_path}")
        
        if verbose:
            print(f'Finding {extension} paths in: {folder_path}')
        
        paths = []
        for entry in sorted(os.scandir(folder_path), key=lambda e: e.name.lower()):
            if entry.is_file() and (entry.name.lower().endswith(extension)):
                paths.append(entry.path)

        if verbose:
            print(f"Found {len(paths)} {extension} files.")
        
        return paths

    @staticmethod
    def get_tiff_paths(folder_path, verbose=True):
        """Finds all TIFF file paths within a specified folder."""
        return FileHandler.get_paths(folder_path = folder_path, 
                                     extension = (".tif", ".tiff"), 
                                     verbose = verbose)
    
    @staticmethod
    def get_npy_paths(folder_path, verbose=True):
        """Finds all NUMPY (NPY) file paths within a specified folder."""
        return FileHandler.get_paths(folder_path = folder_path, 
                                     extension = ".npy", 
                                     verbose = verbose)
    
    @staticmethod
    def extract_folder_path(file_path):
        """
        Extracts the folder path from the given file path.

        Parameters:
        - file_path (str): The complete file path from which to extract the folder path.

        Returns:
        - str: The folder path of the given file path.
        """
        return os.path.dirname(file_path)
    
    @staticmethod
    def find_subfolder(folder_path, pattern, recursive=False, verbose=True):
        """
        Searches for a sub-folder that matches a given pattern within a specified folder path,
        with an option to search recursively or only within the first layer.

        Parameters:
        folder_path (str): The path to the folder in which to search for the sub-folder.
        pattern (str): The pattern to match for the sub-folder name.
        recursive (bool, optional): If True, searches for the pattern recursively. If False, searches only
                                    the first layer of the folder_path. Default is False.
        verbose (bool, optional): If True, prints details about the search process. Default is True.

        Returns:
        str or None: The path to the first sub-folder that matches the pattern, or None if no match is found.
        """
        if not isinstance(folder_path, str):
            raise ValueError("Folder path must be a string.")
        
        if not isinstance(pattern, str):
            raise ValueError("Pattern must be a string.")

        if not os.path.isdir(folder_path):
            raise ValueError(f"The specified folder does not exist: {folder_path}")

        if verbose:
            print(f'Searching for "{pattern}" in: {folder_path}')

        if recursive:
            for root, dirs, _ in os.walk(folder_path):
                for dir_name in dirs:
                    if pattern in dir_name:
                        matched_subfolder_path = os.path.join(root, dir_name)
                        if verbose:
                            print(f'Matched sub-folder found: {matched_subfolder_path}')
                        return matched_subfolder_path
        else:
            # Non-recursive search: Check only the first layer of the directory
            for entry in os.scandir(folder_path):
                if entry.is_dir() and pattern in entry.name:
                    matched_subfolder_path = entry.path
                    if verbose:
                        print(f'Matched sub-folder found: {matched_subfolder_path}')
                    return matched_subfolder_path

        if verbose:
            print("No matched sub-folder found.")
        return None

if __name__ == "__main__":
    print("--- Example usage ---")

    folder_path = "/asap3/petra3/gpfs/p05/2021/data/11008741/processed/wongtakm/datasets/ki4d4e/dvc/volume_960x1280x1280/2021_11008741_syn0154_103L_Mg5Gd_4w_000_fs004/volume0/slices_xyz"
    
    paths = FileHandler.get_tiff_paths(folder_path)

    # folder_path = "/asap3/petra3/gpfs/p05/2021/data/11008741/processed/wongtakm/datasets/ki4d4e/dvc/patch_60x80x80/2021_11008741_syn0154_103L_Mg5Gd_4w_000_fs004/norm_volume0"
    folder_path = "/asap3/petra3/gpfs/p05/2021/data/11008741/processed/wongtakm/datasets/ki4d4e/dvc/volume_960x1280x1280/2021_11008741_syn0154_103L_Mg5Gd_4w_000_fs004/volume0"

    # paths = FileHandler.get_npy_paths(folder_path)
    subfolder_path = FileHandler.find_subfolder(folder_path, pattern = "xyz")

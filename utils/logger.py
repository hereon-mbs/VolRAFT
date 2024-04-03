import sys
import contextlib

import datetime
import logging
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt

from utils.file_handler import FileHandler

class Logger:
    """
    A comprehensive logger that supports redirection of stdout to logging, 
    file logging, and additional functionalities such as saving figures and configurations.
    """
    
    def __init__(self, jobid=None, name='root', level='INFO', time_zone='Europe/Berlin', verbose=False):
        """
        Initializes the logger with a specific job ID, logger name, logging level, time zone, and verbosity setting.

        Parameters:
            jobid (str, optional): Unique identifier for the job.
            name (str): The name of the logger. Defaults to 'root'.
            level (str): The logging level. Defaults to 'INFO'.
            time_zone (str): The time zone to use for timestamps. Defaults to 'Europe/Berlin'.
            verbose (bool): Controls the verbosity of logging output. Defaults to False.
        """
        self.jobid = jobid
        self.verbose = verbose
        self.tz = ZoneInfo(time_zone)
        self.logger = logging.getLogger(name)
        self.name = name
        self.level = getattr(logging, level.upper(), logging.INFO)
        self._setup_logger()

        self.checkpoint_folder_path = None
        self.result_folder_path = None
        self.analysis_folder_path = None
        
        # Redirect stdout to this logger
        self._original_stdout = sys.stdout
        sys.stdout = self
        self._redirect_stdout = contextlib.redirect_stdout(self)

    def __enter__(self):
        """
        Context manager entry point to start redirecting stdout to this logger.
        """
        self._redirect_stdout.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Context manager exit point to stop redirecting stdout and restore the original stdout.
        """
        self._redirect_stdout.__exit__(exc_type, exc_value, traceback)
        sys.stdout = self._original_stdout

    def setup(self):
        """
        Sets up this logger to redirect all stdout output to itself.

        This method captures the current stdout, allowing all subsequent print statements or
        any output to stdout to be redirected to this logger's write method. This is useful
        for capturing and logging output from libraries that use print statements or other
        direct stdout outputs.
        """
        self._stdout = sys.stdout  # Store the original stdout
        sys.stdout = self  # Redirect stdout to this logger instance

    def teardown(self):
        """
        Restores the original stdout, stopping the redirection of stdout output to this logger.

        This method should be called after logging is complete to ensure that stdout behaves
        as expected elsewhere in the application. It is especially important to call this
        method before the program exits or if control is returned to a different part of the
        application that does not expect stdout to be redirected.
        """
        if hasattr(self, '_stdout') and self._stdout:
            sys.stdout = self._stdout  # Restore the original stdout
            self._stdout = None  # Clear the stored stdout to prevent potential circular references

    def print(self, message):
        """
        A convenience method that allows direct printing to this logger, using its write method.

        This method is particularly useful for logging messages in a manner similar to the
        built-in print function, but with the output being captured by the logger's configured
        handlers instead of going directly to stdout.

        Parameters:
            message (str): The message to print/log.
        """
        self.write(message)  # Delegate to the write method to handle the message


    def _setup_logger(self):
        """
        Sets up the logger with basic configuration.
        """
        logging.basicConfig(level=self.level,
                            format='[%(asctime)s] %(message)s',
                            datefmt='%Y/%m/%d %I:%M:%S %p',
                            handlers=[logging.StreamHandler(sys.stderr)])  # Log to stderr

    def log(self, message):
        """
        Logs a message at the configured logging level.

        Parameters:
            message (str): The message to log.
        """
        if message.strip():
            self.logger.log(self.level, message)

    def write(self, message):
        """
        Writes a message to the logger. This method is used to redirect stdout to the logger.

        Parameters:
            message (str): The message to write.
        """
        self.log(message)

    def flush(self):
        """
        Flushes the logging output. This method is needed to comply with the file-like object interface.
        """
        # Implement flushing behavior if necessary, especially when logging to files.

    @staticmethod
    def get_now_time_stamp(time_zone='Europe/Berlin'):
        """
        Returns the current timestamp in a specific time zone, formatted as a string.

        Parameters:
            time_zone (str): The time zone to use. Defaults to 'Europe/Berlin'.

        Returns:
            str: The current timestamp formatted as a string.
        """
        tz = ZoneInfo(time_zone)
        return datetime.datetime.now(tz=tz).strftime('%Y%m%d_%H%M%S_%f')

    @staticmethod
    def get_now_time(time_zone='Europe/Berlin'):
        """
        Returns the current time in a specific time zone, formatted as a string.

        Parameters:
            time_zone (str): The time zone to use. Defaults to 'Europe/Berlin'.

        Returns:
            str: The current time formatted as a string.
        """
        tz = ZoneInfo(time_zone)
        return datetime.datetime.now(tz=tz).strftime('%Y/%m/%d %H:%M:%S %p')

    def set_verbose(self, verbose):
        """
        Adjusts the verbosity of the logger.

        Parameters:
            verbose (bool): If True, enables verbose logging.
        """
        self.verbose = verbose

    def build_output(self, output_folder_path='./results', make_checkpoint=True,
                     make_result=False, make_analysis=False, dataset_name=None):
        """
        Prepares output directories for logging, checkpoints, results, and analysis based on the specified configurations.
        
        Parameters:
            output_folder_path (str): Base path for output folders.
            make_checkpoint (bool): Whether to create a checkpoint subfolder.
            make_result (bool): Whether to create a result subfolder.
            make_analysis (bool): Whether to create an analysis subfolder.
            dataset_name (str, optional): Specific dataset name to include in the path.
        """
        # Ensure the base output folder exists
        FileHandler.mkdir(output_folder_path)

        # Define folder path within the output directory
        # folder_path = os.path.join(output_folder_path, self.name)
        folder_path = FileHandler.join(output_folder_path, self.name)

        # Determine the target folder path
        self.time_stamp = self.get_now_time_stamp()
        target_subpath = self.jobid if self.jobid else self.time_stamp
        target_subpath = FileHandler.join(target_subpath, dataset_name) if dataset_name and (make_analysis or make_result) else target_subpath
        self.target_folder_path = FileHandler.join(folder_path, target_subpath)

        # Create the target folder if it doesn't exist
        # os.makedirs(self.target_folder_path, mode=0o711, exist_ok=True)
        FileHandler.mkdir(self.target_folder_path)

        # Setup logging to a file within the target folder
        self.target_log = FileHandler.join(self.target_folder_path, f'{self.name}_{self.time_stamp}.log')
        self._setup_file_logging(self.target_log)

        # Create specific subfolders if requested
        for folder_type in ['checkpoint', 'result', 'analysis']:
            if locals()[f'make_{folder_type}']:
                setattr(self, f'{folder_type}_folder_path', 
                        FileHandler.join(self.target_folder_path, f'{folder_type}_{self.time_stamp}'))
                FileHandler.mkdir(getattr(self, f'{folder_type}_folder_path'))

        self.log(f'Start to {self.name}')

    def load_output_predict(self, target_folder_path):
        """
        Loads an existing output directory for predictions, initializing logging in the specified path.
        
        Parameters:
            target_folder_path (str): The path to the existing output directory.
        """
        self.time_stamp = self.get_now_time_stamp()
        self.target_folder_path = target_folder_path

        # Prepare the logging file in the existing output directory
        self.target_log = FileHandler.join(self.target_folder_path, f'{self.name}_{self.time_stamp}.log')
        self._setup_file_logging(self.target_log)

        self.log(f'Start {self.name}')

    def _setup_file_logging(self, log_file_path):
        """
        Configures logging to output to a specific file, with the given format and level.
        
        Parameters:
            log_file_path (str): The path to the log file.
        """
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y/%m/%d %I:%M:%S %p')
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        if self.logger.hasHandlers():
            self.logger.handlers.clear()  # Remove existing handlers
        self.logger.addHandler(file_handler)

        # Ensure the logger is set to log INFO messages and above
        self.logger.setLevel(logging.INFO)


    def save_fig(self, filename='log', format='png', folder_path=None, verbose=False):
        """
        Saves the current matplotlib figure to a file.

        Parameters:
            filename (str): Name of the file to save the figure as.
            format (str): Format of the saved figure.
            folder_path (str, optional): Custom folder path to save the figure. Uses checkpoint folder if None.
        """
        folder = self.checkpoint_folder_path if folder_path is None else folder_path
        path = FileHandler.join(folder, f'{filename}.{format}')
        plt.savefig(path)

        if verbose:
            self.log(f'Figure saved to {path}')

    def save_config(self, config_path, format='yaml', verbose=False):
        """
        Copies the configuration file to the checkpoint directory.

        Parameters:
            config_path (str): Path to the configuration file to copy.
            format (str): Format of the configuration file.
        """
        if config_path and FileHandler.has_extension(config_path, ('.yaml', '.yml')):
            dst_path = FileHandler.join(self.checkpoint_folder_path, f'config_{self.get_now_time_stamp()}.{format}')
            FileHandler.cp(src = config_path, dst = dst_path)
            if verbose:
                self.log(f'Configuration file copied to {dst_path}')
        else:
            if verbose:
                self.log(f'Configuration file {config_path} not found or invalid.')

if __name__ == "__main__":
    # Example Usage
    print("--- Example usage ---")
    logger = Logger(jobid = "test", name="test")
    logger.build_output()
    logger.print("this line is called by print()")
    logger.print("this line is called by print() using method {}".format(2))
    
    print("--- Example usage ---")
    with logger:
        print("this line is called by print() in context")
        print("another line is called by print() in context")
    print("third line is called by print() in context")

    print("--- Example usage ---")
    logger.set_verbose(True)
    logger.print("this line is called by print() in verbose mode")
    logger.set_verbose(False)

    print("--- Example usage ---")
    logger.setup()
    print("this line is called by print() in setup mode")
    print("another line is called by print() in setup mode")
    print("third line is called by print() in setup mode")
    logger.teardown()
    print("this line is called by print() after teardown mode")

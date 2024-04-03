import numpy as np
import torch
import matplotlib as mpl

import PIL
import tqdm
import scipy
import yaml

def print_versions() -> None:
    print(f'torch version: {torch.__version__}')
    print(f'numpy version: {np.__version__}')
    print(f'matplotlib version: {mpl.__version__}')
    print(f'PIL version: {PIL.__version__}')
    print(f'tqdm version: {tqdm.__version__}')
    print(f'scipy version: {scipy.__version__}')
    print(f'pyyaml version: {yaml.__version__}')

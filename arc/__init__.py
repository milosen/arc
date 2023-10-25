import os
import warnings
from tqdm import TqdmExperimentalWarning

from arc.definitions import RESULTS_DEFAULT_PATH

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

os.makedirs(RESULTS_DEFAULT_PATH, exist_ok=True)

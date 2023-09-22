import os
import warnings
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

CORPUS_DEFAULT_PATH = os.path.join('resources', 'example_corpus')
RESULTS_DEFAULT_PATH = "arc_results"
os.makedirs(RESULTS_DEFAULT_PATH, exist_ok=True)

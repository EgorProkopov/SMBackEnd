import os

import dotenv


dotenv.load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "VITONHD")

WEIGHTS_URL = os.getenv("WEIGHTS_URL")
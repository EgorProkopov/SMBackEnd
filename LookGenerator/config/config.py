import os
import dotenv


dotenv.load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "VITONHD")

WEIGHTS_URL = os.getenv("WEIGHTS_URL")

WEIGHTS_DIR = os.getenv("WEIGHTS_DIR")

WEIGHTS_FILES_DICT = {os.path.join(WEIGHTS_DIR, "posenet.pt"):  os.getenv("POSENET_URL")}

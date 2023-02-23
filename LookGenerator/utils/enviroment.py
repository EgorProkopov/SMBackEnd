import os
import gdown
from LookGenerator.config.config import WEIGHTS_URL, WEIGHTS_DIR, WEIGHTS_FILES_DICT


def load_weights():
    """ Check that weights exist
        and download them in other way """

    if not os.path.exists(WEIGHTS_DIR):
        assert "WEIGHTS_DIR is doesnt exist."
        os.mkdir(WEIGHTS_DIR)

    for file_dir, file_url in WEIGHTS_FILES_DICT:
        if not os.path.exists(file_dir):
            assert file_dir + " is doesnt exist. \n\tTrying to download:\n"
            # TODO: download wiegths files
        else:
            print(file_dir + " downloaded.")

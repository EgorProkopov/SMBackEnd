import os
import gdown
from LookGenerator.config.config import Config


def load_weights() -> bool:
    """ Check that weights exist
        and download them in other way """

    print(Config.WEIGHTS_DIR, Config.WEIGHTS_URL, Config.POSENET_URL)

    if not os.path.exists(Config.WEIGHTS_DIR):
        assert "WEIGHTS_DIR is doesnt exist."
        os.mkdir(os.path.join(Config.PROJECT_ROOT, Config.WEIGHTS_DIR))

    for file_dir, file_url in Config.WEIGHTS_FILES_DICT.items():
        if not os.path.exists(os.path.join(Config.PROJECT_ROOT, Config.WEIGHTS_DIR, file_dir)):
            assert file_dir + " is doesnt exist. \n\tTrying to download:\n"
            gdown.download(url=file_url, output=Config.WEIGHTS_DIR, fuzzy=True)
        else:
            print(file_dir + " file has already downloaded.")

import os
from typing import get_type_hints, Union
from dotenv import load_dotenv

CONFIG_ENV_FILE = '../../config.env'


def load_env():
    return load_dotenv(CONFIG_ENV_FILE)


class AppConfigError(Exception):
    assert Exception


def _parse_bool(val: Union[str, bool]) -> bool:  # pylint: disable=E1136
    return val if type(val) == bool else val.lower() in ['true', 'yes', '1']


class Config:
    def _load_field(self, field, env):
        if not field.isupper():
            return

        # Raise AppConfigError if required field not supplied
        default_value = getattr(self, field, None)
        if default_value is None and env.get(field) is None:
            raise AppConfigError('The {} field is required'.format(field))

        # Cast env var value to expected type and raise AppConfigError on failure
        try:
            var_type = get_type_hints(self)[field]
            if var_type == bool:
                value = _parse_bool(env.get(field, default_value))
            else:
                value = var_type(env.get(field, default_value))

            self.__setattr__(field, value)

        except ValueError:
            raise AppConfigError('Unable to cast value of "{}" to type "{}" for "{}" field'.format(
                    env[field],
                    var_type,
                    field
                )
            )


class WeightsConfig(Config):
    WEIGHTS_URL: str
    WEIGHTS_DIR: str
    POSENET_URL: str

    """
    Map environment variables to class fields according to these rules:
      - Field won't be parsed unless it has a type annotation
      - Field will be skipped if not in all caps
      - Class field and environment variable name are the same
    """
    def __init__(self, env):
        if not load_env():
            raise AppConfigError('Cannot find "config.env" file. Possibly you should change CONFIG_LOAD_FILE\n')

        for field in self.__annotations__:
            super()._load_field(field, env)

        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
        self.WEIGHTS_FILES_DICT = {'posenet.pt': self.POSENET_URL}

    def __repr__(self):
        return str(self.__dict__)


class DatasetConfig(Config):
    DATASET_DIR: str
    HOUSE_ROOM_DATASET: str

    def __init__(self, env):
        if not load_env():
            raise AppConfigError('Cannot find "config.env" file. Possibly you should change CONFIG_LOAD_FILE\n')

        for field in self.__annotations__:
            super()._load_field(field, env)


if __name__ == "__main__":
    weightConfig = WeightsConfig(os.environ)
    datasetConfig = DatasetConfig(os.environ)


# 3rd party dependencies
import tensorflow

# package dependencies
from deepface.commons.logger import Logger

logger = Logger(module="commons.package_utils")


def get_tf_major_version() -> int:
    """
    Find tensorflow's major version
    Returns
        major_version (int)
    """
    return int(tensorflow.__version__.split(".", maxsplit=1)[0])

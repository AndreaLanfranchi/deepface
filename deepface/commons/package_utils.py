# 3rd party dependencies
import tensorflow

def get_tf_major_version() -> int:
    """
    Find tensorflow's major version
    Returns
        major_version (int)
    """
    return int(tensorflow.__version__.split(".", maxsplit=1)[0])

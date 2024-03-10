from typing import List
import os
import zipfile
import gdown
import numpy
from deepface.commons import package_utils, folder_utils
from deepface.commons.logger import Logger
from deepface.core.decomposer import Decomposer
from deepface.core.types import BoxDimensions

logger = Logger.get_instance()

# --------------------------------
# dependency configuration

tf_version = package_utils.get_tf_major_version()

if tf_version == 1:
    from keras.models import Model, Sequential
    from keras.layers import (
        Convolution2D,
        LocallyConnected2D,
        MaxPooling2D,
        Flatten,
        Dense,
        Dropout,
    )
else:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import (
        Convolution2D,
        LocallyConnected2D,
        MaxPooling2D,
        Flatten,
        Dense,
        Dropout,
    )


# -------------------------------------
# pylint: disable=too-few-public-methods
class DeepFaceClient(Decomposer):
    """
    Fb's DeepFace model class
    """

    def __init__(self):
        self._name = str(__name__.rsplit(".", maxsplit=1)[-1])
        self._input_shape = BoxDimensions(152, 152)
        self._output_shape = 4096
        self._model = load_model()

    def process(self, img: numpy.ndarray) -> List[float]:
        """
        find embeddings with OpenFace model
        Args:
            img (numpy.ndarray): pre-loaded image in BGR
        Returns
            embeddings (list): multi-dimensional vector
        """
        # model.predict causes memory issue when it is called in a for loop
        # embedding = model.predict(img, verbose=0)[0].tolist()
        return self.model(img, training=False).numpy()[0].tolist()


def load_model(
    # pylint: disable=line-too-long
    url="https://github.com/swghosh/DeepFace/releases/download/weights-vggface2-2d-aligned/VGGFace2_DeepFace_weights_val-0.9034.h5.zip",
    # pylint: enable=line-too-long
) -> Model:
    """
    Construct DeepFace model, download its weights and load
    """
    base_model = Sequential()
    base_model.add(
        Convolution2D(
            32, (11, 11), activation="relu", name="C1", input_shape=(152, 152, 3)
        )
    )
    base_model.add(MaxPooling2D(pool_size=3, strides=2, padding="same", name="M2"))
    base_model.add(Convolution2D(16, (9, 9), activation="relu", name="C3"))
    base_model.add(LocallyConnected2D(16, (9, 9), activation="relu", name="L4"))
    base_model.add(
        LocallyConnected2D(16, (7, 7), strides=2, activation="relu", name="L5")
    )
    base_model.add(LocallyConnected2D(16, (5, 5), activation="relu", name="L6"))
    base_model.add(Flatten(name="F0"))
    base_model.add(Dense(4096, activation="relu", name="F7"))
    base_model.add(Dropout(rate=0.5, name="D0"))
    base_model.add(Dense(8631, activation="softmax", name="F8"))

    # ---------------------------------

    file_name = "VGGFace2_DeepFace_weights_val-0.9034.h5"
    weight_file = os.path.join(folder_utils.get_weights_dir(), file_name)

    if os.path.isfile(weight_file) != True:
        logger.info(f"Download : {file_name}")

        source_file = f"{file_name}.zip"
        output = os.path.join(folder_utils.get_weights_dir(), source_file)
        gdown.download(url, output, quiet=False)

        # unzip VGGFace2_DeepFace_weights_val-0.9034.h5.zip
        with zipfile.ZipFile(output, "r") as zip_ref:
            zip_ref.extractall(folder_utils.get_weights_dir())

    base_model.load_weights(weight_file)

    # drop F8 and D0. F7 is the representation layer.
    deepface_model = Model(
        inputs=base_model.layers[0].input, outputs=base_model.layers[-3].output
    )

    return deepface_model

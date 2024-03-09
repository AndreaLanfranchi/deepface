import matplotlib.pyplot as pyplot
import numpy
from deepface import DeepFace
from deepface.modules import verification
from deepface.models.decomposer import Decomposer
from deepface.commons.logger import Logger

logger = Logger.get_instance()

# ----------------------------------------------
# build face recognition model

model_name = "VGG-Face"

model: Decomposer = DeepFace.get_recognition_model(name=model_name)

target_size = model.input_shape

logger.info(f"target_size: {target_size}")

# ----------------------------------------------
# load images and find embeddings

img1 = DeepFace.detect_faces(img_path="dataset/img1.jpg", target_size=target_size)[0]["face"]
img1 = numpy.expand_dims(img1, axis=0)  # to (1, 224, 224, 3)
img1_representation = model.find_embeddings(img1)

img2 = DeepFace.detect_faces(img_path="dataset/img3.jpg", target_size=target_size)[0]["face"]
img2 = numpy.expand_dims(img2, axis=0)
img2_representation = model.find_embeddings(img2)

img1_representation = numpy.array(img1_representation)
img2_representation = numpy.array(img2_representation)

# ----------------------------------------------
# distance between two images - euclidean distance formula
distance_vector = numpy.square(img1_representation - img2_representation)
current_distance = numpy.sqrt(distance_vector.sum())
logger.info(f"Euclidean distance: {current_distance}")

threshold = verification.find_threshold(model_name=model_name, distance_metric="euclidean")
logger.info(f"Threshold for {model_name}-euclidean pair is {threshold}")

if current_distance < threshold:
    logger.info(
        f"This pair is same person because its distance {current_distance}"
        f" is less than threshold {threshold}"
    )
else:
    logger.info(
        f"This pair is different persons because its distance {current_distance}"
        f" is greater than threshold {threshold}"
    )
# ----------------------------------------------
# expand vectors to be shown better in graph

img1_graph = []
img2_graph = []
distance_graph = []

for i in range(0, 200):
    img1_graph.append(img1_representation)
    img2_graph.append(img2_representation)
    distance_graph.append(distance_vector)

img1_graph = numpy.array(img1_graph)
img2_graph = numpy.array(img2_graph)
distance_graph = numpy.array(distance_graph)

# ----------------------------------------------
# plotting

fig = pyplot.figure()

ax1 = fig.add_subplot(3, 2, 1)
pyplot.imshow(img1[0])
pyplot.axis("off")

ax2 = fig.add_subplot(3, 2, 2)
im = pyplot.imshow(img1_graph, interpolation="nearest", cmap=pyplot.cm.ocean)
pyplot.colorbar()

ax3 = fig.add_subplot(3, 2, 3)
pyplot.imshow(img2[0])
pyplot.axis("off")

ax4 = fig.add_subplot(3, 2, 4)
im = pyplot.imshow(img2_graph, interpolation="nearest", cmap=pyplot.cm.ocean)
pyplot.colorbar()

ax5 = fig.add_subplot(3, 2, 5)
pyplot.text(0.35, 0, f"Distance: {current_distance}")
pyplot.axis("off")

ax6 = fig.add_subplot(3, 2, 6)
im = pyplot.imshow(distance_graph, interpolation="nearest", cmap=pyplot.cm.ocean)
pyplot.colorbar()

pyplot.show()

# ----------------------------------------------

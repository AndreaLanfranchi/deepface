import matplotlib.pyplot as pyplot
from deepface import DeepFace
from deepface.commons.logger import Logger

logger = Logger.get_instance()

# some models (e.g. Dlib) and detectors (e.g. retinaface) do not have test cases
# because they require to install huge packages
# this module is for local runs

model_names = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "Dlib",
    "ArcFace",
    "SFace",
]

detector_backends = [
    "opencv",
    "ssd",
    "dlib",
    "mtcnn",
    # "mediapipe", # crashed in mac
    "retinaface",
    "yunet",
    "yolov8",
]


# verification
for model_name in model_names:
    obj = DeepFace.verify(
        img1_path="dataset/img1.jpg", img2_path="dataset/img2.jpg", decomposer=model_name
    )
    logger.info(obj)
    logger.info("---------------------")

# represent
for model_name in model_names:
    embedding_objs = DeepFace.represent(img_path="dataset/img1.jpg", decomposer=model_name)
    for embedding_obj in embedding_objs:
        embedding = embedding_obj["embedding"]
        logger.info(f"{model_name} produced {len(embedding)}D vector")


# find
dfs = DeepFace.find(
    img="dataset/img1.jpg", db_path="dataset", extractor="Facenet", detector="mtcnn"
)
for df in dfs:
    logger.info(df)


# img_paths = ["dataset/img11.jpg", "dataset/img11_reflection.jpg", "dataset/couple.jpg"]
img_paths = ["dataset/img11.jpg"]
for img_path in img_paths:
    # extract faces
    for detector_backend in detector_backends:
        face_objs = DeepFace.detect_faces(
            img_path=img_path,
            detector=detector_backend,
            align=True,
        )
        for face_obj in face_objs:
            face = face_obj["face"]
            logger.info(detector_backend)
            logger.info(face_obj["facial_area"])
            logger.info(face_obj["confidence"])

            # we know opencv sometimes cannot find eyes
            if face_obj["facial_area"]["left_eye"] is not None:
                assert isinstance(face_obj["facial_area"]["left_eye"], tuple)
                assert isinstance(face_obj["facial_area"]["left_eye"][0], int)
                assert isinstance(face_obj["facial_area"]["left_eye"][1], int)

            if face_obj["facial_area"]["right_eye"] is not None:
                assert isinstance(face_obj["facial_area"]["right_eye"], tuple)
                assert isinstance(face_obj["facial_area"]["right_eye"][0], int)
                assert isinstance(face_obj["facial_area"]["right_eye"][1], int)

            assert isinstance(face_obj["confidence"], float)
            assert face_obj["confidence"] <= 1

            pyplot.imshow(face)
            pyplot.axis("off")
            pyplot.show()
            logger.info("-----------")

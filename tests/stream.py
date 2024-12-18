# Sample code to run the stream function using a network camera
# DeepFace.stream(db_path="dataset",
#                 model_name="VGG-Face",
#                 faces_count_threshold = 2,
#                 freeze_time_seconds = 3,
#                 valid_frames_count = 3,
#                 source = "rtsp://admin:123Pa$$word!@192.168.1.33/profile3/media.smp"
#                 )

# Sample code to run the stream function using onboard camera
# from deepface import DeepFace
# DeepFace.stream(
#     db_path="dataset",
#     detector="fastmtcnn",
#     extractor="openface",
#     faces_count_threshold=2,
#     freeze_time_seconds=2,
#     valid_frames_count=2,
#     source=0,  # 0 for onboard camera
# )
# exit()

# from deepface.core.analyzer import Analyzer
import os
import time
import warnings


warnings.filterwarnings("ignore")
from deepface.infra import analysis, detection, extraction, continuos_detection

# age_analyzer: Analyzer = Analyzer.instance("age")
# gender_analyzer: Analyzer = Analyzer.instance("gender")
# results_list = detection.batch_detect_faces(inputs=".", detector="fastmtcnn")
# total_faces = int(0)
# for results in results_list:
#     # if 0 != len(results):
#     #     continue
#     print(f"Detected {len(results)} face(s) in {results.tag}")
#     i:int = 0
#     for detection in results.detections:
#         age_results = age_analyzer.process(results.img, detection)
#         gender_results = gender_analyzer.process(results.img, detection)
#         print(f"Face {i} attributes: {gender_results.value} ({age_results.value})")
#         # print(attr_results)
#         # print(f"Face detected at {detection.bounding_box.xywh} with confidence {detection.confidence}")
#     total_faces += len(results)
# print(f"Total faces detected: {total_faces}")

# r = extraction.batch_extract_faces(inputs=r".\dataset", detector="yunet")
# for tag, detections in r.items():
#     count: int = 0
#     if detections is not None:
#         count = len(detections)
#     print(f"{tag}: {count} face(s) detected")

# r = detection.batch_detect_faces(
#     inputs=r".\dataset\couple.jpg",
#     detector="fastmtcnn",
#     extractor="default",
#     attributes=["age", "gender"],
#     raise_notfound=False,
# )
# print(r)

# r = extraction.batch_extract_faces(
#     inputs=r".\dataset\couple.jpg",
#     detector="fastmtcnn",
#     key_points=True,
#     raise_notfound=False,
# )
# print(r)

# r = analysis.batch_analyze_faces(
#     inputs=r".\dataset\couple.JPG",
#     detector="fastmtcnn",
#     attributes=None,
#     key_points=True,
#     raise_notfound=False,
# )
# print(r)

o = continuos_detection.ContinuosFaceDetection(os.path.join(os.getcwd(), "dataset"), extractor="default", attributes=["gender"])
o.start()
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    o.stop()

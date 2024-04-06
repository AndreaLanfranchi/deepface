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
#     detector="mtcnn",
#     extractor="openface",
#     faces_count_threshold=2,
#     freeze_time_seconds=2,
#     valid_frames_count=2,
#     source=0,  # 0 for onboard camera
# )

from deepface.infra import (
    detection,
    extraction,
)

# results_list = detection.batch_detect_faces(inputs=".", detector="yunet")
# total_faces = int(0)
# for results in results_list:
#     print(f"Detected {len(results)} face(s) in {results.tag}")
#     for detection in results.detections:
#         print(f"Face detected at {detection.bounding_box.xywh} with confidence {detection.confidence}")
#     total_faces += len(results)

# print(f"Total faces detected: {total_faces}")

r = extraction.batch_extract_faces(inputs=".\\dataset")
for tag, detections in r.items():
    count: int = 0
    if detections is not None:
        count = len(detections)
    print(f"{tag}: {count} face(s) detected")

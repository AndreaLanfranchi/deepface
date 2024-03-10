from deepface import DeepFace

# Sample code to run the stream function using a network camera
# DeepFace.stream(db_path="dataset",
#                 model_name="VGG-Face",
#                 faces_count_threshold = 2,
#                 freeze_time_seconds = 3,
#                 valid_frames_count = 3,
#                 source = "rtsp://admin:123Pa$$word!@192.168.1.33/profile3/media.smp"
#                 )

# Sample code to run the stream function using onboard camera
DeepFace.stream(db_path="dataset",
                decomposer="fbdeepface",
                faces_count_threshold = 2,
                freeze_time_seconds = 2,
                valid_frames_count = 5,
                source = 0 # 0 for onboard camera
                )

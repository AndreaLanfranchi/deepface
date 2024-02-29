from deepface import DeepFace

DeepFace.stream("dataset", model_name="VGG-Face", faces_count_threshold=1) #opencv
#DeepFace.stream("dataset", detector_backend = 'opencv')
#DeepFace.stream("dataset", detector_backend = 'ssd')
#DeepFace.stream("dataset", detector_backend = 'mtcnn')
#DeepFace.stream("dataset", detector_backend = 'dlib')
#DeepFace.stream("dataset", detector_backend = 'retinaface')

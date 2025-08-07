import cv2
from insightface.app import FaceAnalysis
from src.logging.logger import logger

class FaceDetector:
    def __init__(self):
        try:
            self.face_analyzer = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("Face detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {str(e)}")
            raise

    def detect_and_crop(self, image):
        try:
            faces = self.face_analyzer.get(image)
            if not faces:
                logger.warning("No faces detected in image")
                return None
                
            logger.debug(f"Detected {len(faces)} faces")
            return [self._process_face(face, image) for face in faces]
            
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            raise

    def _process_face(self, face, image):
        try:
            x1, y1, x2, y2 = map(int, face.bbox)
            cropped = image[y1:y2, x1:x2]
            logger.debug(f"Processed face at coordinates: {x1},{y1} - {x2},{y2}")
            return cv2.resize(cropped, (224, 224))
        except Exception as e:
            logger.error(f"Face processing failed: {str(e)}")
            raise

import mediapipe as mp
import numpy as np
from src.logging.logger import logger

class LandmarkExtractor:
    def __init__(self):
        try:
            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, 
                max_num_faces=1,
                min_detection_confidence=0.5
            )
            logger.info("Landmark extractor initialized")
        except Exception as e:
            logger.error(f"Landmark extractor init failed: {str(e)}")
            raise

    def extract(self, image):
        try:
            results = self.mp_face_mesh.process(image)
            if not results.multi_face_landmarks:
                logger.warning("No landmarks detected")
                return None
                
            landmarks = results.multi_face_landmarks[0].landmark
            logger.debug("Successfully extracted landmarks")
            return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten().astype(np.float32)
            
        except Exception as e:
            logger.error(f"Landmark extraction failed: {str(e)}")
            raise

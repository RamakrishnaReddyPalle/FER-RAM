import torch
from src.logging.logger import logger
from src.components import FaceDetector, LandmarkExtractor, HybridEmotionModel
from src.utils.image_utils import preprocess_image

class ExpressionPredictor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        try:
            self.model = self._load_model(model_path)
            self.face_detector = FaceDetector()
            self.landmark_extractor = LandmarkExtractor()
            self.class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
            logger.info("Predictor initialized successfully")
        except Exception as e:
            logger.error(f"Predictor initialization failed: {str(e)}")
            raise

    def _load_model(self, model_path):
        try:
            model = HybridEmotionModel()
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Loaded model from {model_path}")
            return model.eval().to(self.device)
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

    def predict(self, image):
        try:
            cropped_faces = self.face_detector.detect_and_crop(image)
            if not cropped_faces:
                logger.info("No faces found for prediction")
                return []

            predictions = []
            logger.debug(f"Processing {len(cropped_faces)} faces")
            
            for idx, face in enumerate(cropped_faces):
                try:
                    img_tensor = preprocess_image(face).to(self.device)
                    landmarks = self.landmark_extractor.extract(face)
                    
                    if landmarks is None:
                        logger.warning(f"Face {idx+1}: No landmarks detected")
                        continue

                    lm_tensor = torch.from_numpy(landmarks).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        logits, *_ = self.model(img_tensor, lm_tensor)
                        prob = torch.softmax(logits, dim=1)
                        confidence, pred_idx = torch.max(prob, dim=1)

                    predictions.append({
                        'emotion': self.class_names[pred_idx.item()],
                        'confidence': confidence.item(),
                        'face': face
                    })
                    logger.debug(f"Face {idx+1} prediction: {predictions[-1]}")
                    
                except Exception as e:
                    logger.error(f"Face {idx+1} processing failed: {str(e)}")
                    continue
                    
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

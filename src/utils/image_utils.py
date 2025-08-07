from PIL import Image
import torchvision.transforms as transforms

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    return transform(Image.fromarray(image)).unsqueeze(0)

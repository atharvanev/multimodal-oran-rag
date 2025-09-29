import base64
from io import BytesIO
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch


class BlipCaptioner:
    """
    A class for generating image captions using BLIP model.
    Supports both conditional and unconditional captioning.
    """
    
    def __init__(self, model_name="Salesforce/blip-image-captioning-large", device=None):
        """
        Initialize the captioner with a BLIP model.
        
        Args:
            model_name (str): Hugging Face model identifier
            device (str): Device to run on ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model on {self.device}...")
        
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
        print("Model loaded successfully!")
    
    def _decode_base64_image(self, base64_string):
        """
        Decode a base64 string to a PIL Image.
        
        Args:
            base64_string (str): Base64 encoded image string
            
        Returns:
            PIL.Image: RGB image
        """
        # Remove data URI scheme if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        return image
    
    def generate_caption(self, base64_image, conditional_text=None, max_length=50, num_beams=4):
        """
        Generate a caption for a base64 encoded image.
        
        Args:
            base64_image (str): Base64 encoded image string
            conditional_text (str, optional): Text prompt for conditional captioning
            max_length (int): Maximum length of generated caption
            num_beams (int): Number of beams for beam search
            
        Returns:
            str: Generated caption
        """
        # Decode image
        image = self._decode_base64_image(base64_image)
        
        # Process inputs
        if conditional_text:
            inputs = self.processor(image, conditional_text, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        # Generate caption
        out = self.model.generate(**inputs, max_length=max_length, num_beams=num_beams)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        return caption

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from .config import Config
from gtts import gTTS
from deep_translator import GoogleTranslator
import os

class DARA:
    def __init__(self, model_id=None):
        self.model_id = model_id or Config.MODEL_ID
        print(f"Loading DARA-Lite ({self.model_id})...")
        self.device = Config.DEVICE
        self.torch_dtype = Config.torch_dtype
        
        # Load Model & Processor
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            torch_dtype=self.torch_dtype, 
            trust_remote_code=True,
            attn_implementation="eager"
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, 
            trust_remote_code=True
        )
        print("DARA-Lite loaded successfully!")

    @torch.inference_mode()
    def detect(self, image_input, mode=Config.MODE_SCENE, language="en"):
        """
        Detects and assists based on the selected mode.
        Args:
            image_input (str or PIL.Image.Image): Path to image or PIL Image object.
            mode (str): Detection mode.
            language (str): Output language code ('en' or 'id').
        """
        if isinstance(image_input, str):
            image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError("Invalid image input. Must be a path or PIL Image.")

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Select Prompt based on Mode
        # Select Prompt based on Mode
        task_prompt = Config.PROMPTS.get(mode, "<OD>")
        
        # Note: We rely on _process_output to refine the raw model result
        # based on the specific mode (e.g., extracting emotion from a caption).

        # Inference
        inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(self.device, self.torch_dtype)
        
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=1,
            use_cache=False,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        # print(f"DEBUG: Generated Text: {generated_text}")
        
        try:
            parsed_answer = self.processor.post_process_generation(
                generated_text, 
                task=task_prompt, 
                image_size=(image.width, image.height)
            )
        except Exception as e:
            # print(f"DEBUG: Post-processing failed: {e}")
            parsed_answer = {task_prompt: generated_text} # Fallback

        # Post-processing / "Smart" Assist Logic
        final_output = self._process_output(parsed_answer, mode, task_prompt, language)
        
        # Generate Audio
        audio_path = self._generate_audio(final_output, language)
        
        return {
            "mode": mode,
            "result": final_output,
            "audio": audio_path,
            "language": language
        }

    def _process_output(self, parsed_answer, mode, task_prompt, language="en"):
        """
        Refines the raw model output into helpful advice.
        """
        raw_text = parsed_answer.get(task_prompt, "")
        
        # If raw_text is a dictionary (like OD), convert to string
        if isinstance(raw_text, dict):
            raw_text = str(raw_text)

        if mode == Config.MODE_SCENE:
            # Clean up the text
            text = self._clean_text(raw_text)
            if language == "id":
                text = GoogleTranslator(source='auto', target='id').translate(text)
            return text
        
        elif mode == Config.MODE_EMOTION:
            # Enhanced keyword matching
            lower_text = str(raw_text).lower()
            emotion = "Neutral"
            advice = "Ask how they are doing."
            
            if any(x in lower_text for x in ["smile", "happy", "laugh", "joy"]):
                emotion = "Happy"
                advice = "They seem in good spirits!"
            elif any(x in lower_text for x in ["sad", "cry", "tear", "upset", "frown"]):
                emotion = "Sad"
                advice = "Offer comfort or support."
            elif any(x in lower_text for x in ["angry", "mad", "furious", "shout"]):
                emotion = "Angry"
                advice = "Give them space or ask calmly."
            elif any(x in lower_text for x in ["fear", "scared", "afraid", "terror"]):
                emotion = "Fearful"
                advice = "Reassure them that they are safe."
                
            text = f"{emotion}. {advice}"
            if language == "id":
                text = GoogleTranslator(source='auto', target='id').translate(text)
            return text

        elif mode == Config.MODE_MEDICINE:
            # Regex to find dosage (mg/ml) and potential medicine names
            import re
            dosage_pattern = r"(\d+\s*(?:mg|ml|g|mcg))"
            dosages = re.findall(dosage_pattern, raw_text, re.IGNORECASE)
            
            advice = "Consult a doctor for exact dosage."
            if dosages:
                text = f"Dosage: {', '.join(dosages)}. Take as prescribed."
            else:
                text = self._clean_text(raw_text)

            if language == "id":
                text = GoogleTranslator(source='auto', target='id').translate(text)
            return text

        elif mode == Config.MODE_CURRENCY:
            # Regex to find currency values
            import re
            # Matches Rp 50.000, 50,000, $10, etc.
            currency_pattern = r"(?:Rp|USD|\$|€|£)\s*[\d,.]+|[\d,.]+\s*(?:Rupiah|Dollar|Euro)"
            values = re.findall(currency_pattern, raw_text, re.IGNORECASE)
            
            if values:
                text = f"Currency: {', '.join(values)}."
            else:
                text = "Currency value unclear."
            
            if language == "id":
                text = GoogleTranslator(source='auto', target='id').translate(text)
            return text

        elif mode == Config.MODE_TEXT:
            text = self._clean_text(raw_text)
            if language == "id":
                text = GoogleTranslator(source='auto', target='id').translate(text)
            return text

        return str(raw_text)

    def _generate_audio(self, text, language="en"):
        """
        Generates TTS audio file.
        """
        try:
            tts = gTTS(text=text, lang=language) 
            # Use a unique filename to avoid collisions in multi-user envs
            import uuid
            save_path = f"output_{uuid.uuid4().hex}.mp3"
            tts.save(save_path)
            return save_path
        except Exception as e:
            print(f"TTS Error: {e}")
            return None

    def _clean_text(self, text):
        """
        Removes special tokens and extra whitespace.
        """
        text = str(text)
        text = text.replace("</s>", "").replace("<s>", "")
        return text.strip()

if __name__ == "__main__":
    # Test run
    dara = DARA()
    print("Model initialized.")

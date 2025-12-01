import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from config import Config
from gtts import gTTS
import os

class DARA:
    def __init__(self):
        print(f"Loading DARA-Lite ({Config.MODEL_ID})...")
        self.device = Config.DEVICE
        self.torch_dtype = Config.torch_dtype
        
        # Load Model & Processor
        self.model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_ID, 
            torch_dtype=self.torch_dtype, 
            trust_remote_code=True,
            attn_implementation="eager"
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            Config.MODEL_ID, 
            trust_remote_code=True
        )
        print("DARA-Lite loaded successfully!")

    def detect(self, image_path, mode=Config.MODE_SCENE):
        """
        Detects and assists based on the selected mode.
        """
        image = Image.open(image_path)
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
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        print(f"DEBUG: Generated Text: {generated_text}")
        
        try:
            parsed_answer = self.processor.post_process_generation(
                generated_text, 
                task=task_prompt, 
                image_size=(image.width, image.height)
            )
        except Exception as e:
            print(f"DEBUG: Post-processing failed: {e}")
            parsed_answer = {task_prompt: generated_text} # Fallback

        # Post-processing / "Smart" Assist Logic
        final_output = self._process_output(parsed_answer, mode, task_prompt)
        
        # Generate Audio
        audio_path = self._generate_audio(final_output)
        
        return {
            "mode": mode,
            "result": final_output,
            "audio": audio_path
        }

    def _process_output(self, parsed_answer, mode, task_prompt):
        """
        Refines the raw model output into helpful advice.
        """
        raw_text = parsed_answer.get(task_prompt, "")
        
        # If raw_text is a dictionary (like OD), convert to string
        if isinstance(raw_text, dict):
            raw_text = str(raw_text)

        if mode == Config.MODE_SCENE:
            return f"Scene Description: {raw_text}"
        
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
                
            return f"Emotion: {emotion}. Advice: {advice}. (Context: {raw_text})"

        elif mode == Config.MODE_MEDICINE:
            # Regex to find dosage (mg/ml) and potential medicine names
            import re
            dosage_pattern = r"(\d+\s*(?:mg|ml|g|mcg))"
            dosages = re.findall(dosage_pattern, raw_text, re.IGNORECASE)
            
            advice = "Consult a doctor for exact dosage."
            if dosages:
                advice = f"Detected dosage: {', '.join(dosages)}. Take as prescribed."
            
            return f"Label Read: {raw_text}. \n[DARA Assist]: {advice}"

        elif mode == Config.MODE_CURRENCY:
            # Regex to find currency values
            import re
            # Matches Rp 50.000, 50,000, $10, etc.
            currency_pattern = r"(?:Rp|USD|\$|€|£)\s*[\d,.]+|[\d,.]+\s*(?:Rupiah|Dollar|Euro)"
            values = re.findall(currency_pattern, raw_text, re.IGNORECASE)
            
            if values:
                return f"Currency Detected: {', '.join(values)}. \n[DARA Assist]: Please verify by touch."
            else:
                return f"Currency Description: {raw_text}. \n[DARA Assist]: Specific value not clearly detected."

        elif mode == Config.MODE_TEXT:
            return f"Text Read: {raw_text}"

        return str(raw_text)

    def _generate_audio(self, text):
        """
        Generates TTS audio file.
        """
        try:
            tts = gTTS(text=text, lang='en') # Default to English for now, can switch to 'id'
            save_path = "output.mp3"
            tts.save(save_path)
            return save_path
        except Exception as e:
            print(f"TTS Error: {e}")
            return None

if __name__ == "__main__":
    # Test run
    dara = DARA()
    print("Model initialized.")

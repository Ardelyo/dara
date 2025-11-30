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
            trust_remote_code=True
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
        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(image.width, image.height)
        )

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
            # Simple keyword matching for demo purposes
            lower_text = str(raw_text).lower()
            emotion = "Neutral"
            advice = "Ask how they are doing."
            if "smile" in lower_text or "happy" in lower_text:
                emotion = "Happy"
                advice = "They seem in good spirits!"
            elif "sad" in lower_text or "crying" in lower_text:
                emotion = "Sad"
                advice = "Offer comfort or support."
            elif "angry" in lower_text:
                emotion = "Angry"
                advice = "Give them space or ask calmly."
            return f"Emotion: {emotion}. Advice: {advice}. (Context: {raw_text})"

        elif mode == Config.MODE_MEDICINE:
            # Mock database lookup
            return f"Label Read: {raw_text}. \n[DARA Assist]: If this is Paracetamol, take 500mg after food. Consult a doctor for dosage."

        elif mode == Config.MODE_CURRENCY:
            return f"Currency Detected: {raw_text}. \n[DARA Assist]: Please verify the denomination by touch if possible."

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

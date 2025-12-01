import gradio as gr
import torch
from PIL import Image
import sys
import os

# Robust import logic for 'dara' package
# 1. Try importing directly (if 'dara' is in root or installed)
# 2. Try adding 'src' to path (standard structure)
try:
    from dara.model import DARA
    from dara.config import Config
except ImportError:
    # Try adding src to path
    src_path = os.path.join(os.path.dirname(__file__), "src")
    if os.path.exists(src_path):
        sys.path.insert(0, src_path)
        try:
            from dara.model import DARA
            from dara.config import Config
        except ImportError as e:
            raise ImportError(f"Found 'src' directory but failed to import 'dara': {e}")
    else:
        # Critical error: src missing
        raise ImportError(
            f"CRITICAL: Could not find 'dara' module and 'src' directory is missing at {src_path}. "
            "Please ensure you have uploaded the entire 'src' folder to the Hugging Face Space."
        )

# Initialize Model
print("Initializing DARA Model...")
try:
    dara_model = DARA()
except Exception as e:
    print(f"Error initializing model: {e}")
    dara_model = None

def predict(image, mode, language):
    """
    Main prediction function for Gradio.
    """
    if dara_model is None:
        return "Error: Model not loaded.", None

    if image is None:
        return "Please upload an image.", None

    try:
        # Pass PIL image directly to the model
        
        # Map UI mode names to Config modes
        mode_map = {
            "Scene Description": Config.MODE_SCENE,
            "Emotion Analysis": Config.MODE_EMOTION,
            "Medicine Helper": Config.MODE_MEDICINE,
            "Currency Helper": Config.MODE_CURRENCY,
            "Text Extraction": Config.MODE_TEXT
        }
        
        selected_mode = mode_map.get(mode, Config.MODE_SCENE)
        
        # Map UI language names to codes
        lang_map = {
            "English": "en",
            "Indonesian": "id"
        }
        selected_lang = lang_map.get(language, "en")

        # Run detection
        result = dara_model.detect(image, mode=selected_mode, language=selected_lang)
        
        text_output = result["result"]
        audio_path = result["audio"]
        
        return text_output, audio_path

    except Exception as e:
        return f"An error occurred: {str(e)}", None

# UI Design
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
)

with gr.Blocks(theme=theme, title="DARA - Digital Assistant for Real-world Accessibility") as demo:
    gr.Markdown(
        """
        # 👁️ DARA: Digital Assistant for Real-world Accessibility
        **Empowering visually impaired users with AI-driven scene understanding.**
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Image")
            
            with gr.Group():
                mode_radio = gr.Radio(
                    choices=["Scene Description", "Emotion Analysis", "Medicine Helper", "Currency Helper", "Text Extraction"],
                    value="Scene Description",
                    label="Select Mode / Pilih Mode"
                )
                lang_radio = gr.Radio(
                    choices=["English", "Indonesian"],
                    value="English",
                    label="Select Language / Pilih Bahasa"
                )
            
            submit_btn = gr.Button("Analyze / Analisa", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(label="Result / Hasil", lines=5, show_copy_button=True)
            output_audio = gr.Audio(label="Audio Response", type="filepath")
            
    # Footer / Info
    with gr.Accordion("About DARA", open=False):
        gr.Markdown(
            """
            DARA (Digital Assistant for Real-world Accessibility) is a multimodal AI designed to assist visually impaired individuals.
            It uses the Florence-2 base model for vision-language tasks and provides audio feedback.
            
            **Modes:**
            - **Scene Description**: General description of the image.
            - **Emotion Analysis**: Detects emotions from facial expressions.
            - **Medicine Helper**: Reads medicine labels and dosages.
            - **Currency Helper**: Identifies currency values.
            - **Text Extraction**: Reads any text found in the image.
            """
        )

    submit_btn.click(
        fn=predict,
        inputs=[input_image, mode_radio, lang_radio],
        outputs=[output_text, output_audio]
    )

if __name__ == "__main__":
    demo.launch()

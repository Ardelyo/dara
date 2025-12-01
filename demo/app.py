import gradio as gr
from dara import DARA, Config
import os
from PIL import Image
import traceback

# --- Configuration & Initialization ---
print("Initializing DARA...")
try:
    dara = DARA()
    print("DARA Initialized Successfully!")
except Exception as e:
    print(f"Error initializing DARA: {e}")
    dara = None

# --- Helper Functions ---

def process_image(image, mode, language):
    """
    Processes the image using DARA model based on selected mode and language.
    """
    if dara is None:
        return "Error: Model not initialized. Please check logs.", None

    if image is None:
        return "Please upload an image first.", None
    
    try:
        # Map language name to code
        lang_code = "id" if language == "Indonesian (Bahasa Indonesia)" else "en"
        
        # Save temp image for the model (model expects a path)
        temp_path = "temp_input.jpg"
        image.save(temp_path)
        
        # Run detection
        result = dara.detect(temp_path, mode, language=lang_code)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return result["result"], result["audio"]

    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        print(traceback.format_exc())
        return error_msg, None

# --- UI Layout ---

# Custom Theme
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

with gr.Blocks(theme=theme, title="🪔 DARA - Detect & Assist Recognition AI") as demo:
    
    # Header
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("""
            # 🪔 DARA
            ### Detect & Assist Recognition AI
            *"Mata untuk semua" (Eyes for everyone)*
            """)
        with gr.Column(scale=1):
            gr.Markdown("""
            **DARA** is a Vision-Language Model designed to assist visually impaired users by describing scenes, reading text, identifying currency, and recognizing emotions.
            """)

    # Main Content
    with gr.Row():
        # Left Column: Input
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Input Image", sources=["upload", "clipboard", "webcam"])
            
            with gr.Row():
                mode_dropdown = gr.Dropdown(
                    choices=[
                        Config.MODE_SCENE, 
                        Config.MODE_EMOTION, 
                        Config.MODE_MEDICINE, 
                        Config.MODE_CURRENCY, 
                        Config.MODE_TEXT
                    ],
                    value=Config.MODE_SCENE,
                    label="Select Mode",
                    info="Choose what you want DARA to look for."
                )
                language_dropdown = gr.Dropdown(
                    choices=["English", "Indonesian (Bahasa Indonesia)"],
                    value="English",
                    label="Language",
                    info="Select output language."
                )
            
            submit_btn = gr.Button("👁️ Detect & Assist", variant="primary", size="lg")

        # Right Column: Output
        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="DARA Output", 
                placeholder="Result will appear here...",
                lines=4,
                show_copy_button=True
            )
            output_audio = gr.Audio(label="Voice Assist", type="filepath", autoplay=True)

    # Examples
    gr.Markdown("### 📸 Try with Examples")
    gr.Examples(
        examples=[
            ["sampleimages/food table.jpg", Config.MODE_SCENE, "English"],
            ["sampleimages/sad person.jpg", Config.MODE_EMOTION, "English"],
            ["sampleimages/medic.jpg", Config.MODE_MEDICINE, "English"],
            ["sampleimages/park signs.jpg", Config.MODE_TEXT, "English"],
        ],
        inputs=[input_image, mode_dropdown, language_dropdown],
        outputs=[output_text, output_audio],
        fn=process_image,
        cache_examples=False, # Disable caching to save time/space
        label="Click on an example to load it"
    )

    # Footer
    gr.Markdown("""
    ---
    <div style="text-align: center; opacity: 0.7;">
        <p>Powered by <b>DARA-Lite</b> (Florence-2 Base) | Developed for the DARA Project</p>
        <p><i>Note: This is a demo. Medical and currency advice should be verified.</i></p>
    </div>
    """)

    # Event Listener
    submit_btn.click(
        fn=process_image,
        inputs=[input_image, mode_dropdown, language_dropdown],
        outputs=[output_text, output_audio]
    )

if __name__ == "__main__":
    demo.launch(share=False)

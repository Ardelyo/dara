import gradio as gr
from dara import DARA, Config
import os

# Initialize Model
dara = DARA()

def process_image(image, mode):
    if image is None:
        return "Please upload an image.", None
    
    # Save temp image
    temp_path = "temp_input.jpg"
    image.save(temp_path)
    
    result = dara.detect(temp_path, mode)
    
    return result["result"], result["audio"]

# UI
with gr.Blocks(title="🪔 DARA - Detect & Assist Recognition AI") as demo:
    gr.Markdown("""
    # 🪔 DARA — Full Concept Demo
    > **Detect & Assist Recognition AI**  
    > "Mata untuk semua"
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
            mode_dropdown = gr.Dropdown(
                choices=[
                    Config.MODE_SCENE, 
                    Config.MODE_EMOTION, 
                    Config.MODE_MEDICINE, 
                    Config.MODE_CURRENCY, 
                    Config.MODE_TEXT
                ],
                value=Config.MODE_SCENE,
                label="Select Mode"
            )
            submit_btn = gr.Button("👁️ Detect & Assist", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(label="DARA Output")
            output_audio = gr.Audio(label="Voice Assist")
    
    submit_btn.click(
        fn=process_image,
        inputs=[input_image, mode_dropdown],
        outputs=[output_text, output_audio]
    )

    gr.Markdown("--- \n*Powered by DARA-Lite (Florence-2)*")

if __name__ == "__main__":
    demo.launch(share=False)

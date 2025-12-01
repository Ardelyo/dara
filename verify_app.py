from model import DARA, Config
import os

def verify_app():
    print("Initializing DARA for Inference Verification...")
    dara = DARA()
    
    # Test Images (we created these earlier)
    test_images = {
        Config.MODE_SCENE: "data/images/sample_scene.jpg",
        Config.MODE_MEDICINE: "data/images/sample_medicine.jpg",
        Config.MODE_CURRENCY: "data/images/sample_currency.jpg",
        Config.MODE_EMOTION: "data/images/sample_scene.jpg", # Reuse scene image
        Config.MODE_TEXT: "data/images/sample_medicine.jpg" # Reuse medicine image
    }
    
    print("\n--- Testing Inference Modes ---")
    for mode, img_path in test_images.items():
        print(f"\nTesting Mode: {mode}")
        if os.path.exists(img_path):
            try:
                # We mock the model generation output by overriding _process_output input 
                # because the base model won't produce perfect OCR/Caption for blank images.
                # However, we want to test the LOGIC in _process_output.
                
                # Run actual detection (will produce garbage text for blank images)
                result = dara.detect(img_path, mode)
                print(f"Raw Result: {result['result']}")
                
                # Test Logic with Mocked Text
                mock_text = ""
                if mode == Config.MODE_MEDICINE:
                    mock_text = "Paracetamol 500mg tablet"
                elif mode == Config.MODE_CURRENCY:
                    mock_text = "Rp 50.000"
                elif mode == Config.MODE_EMOTION:
                    mock_text = "A happy person smiling"
                
                if mock_text:
                    processed = dara._process_output({Config.PROMPTS[mode]: mock_text}, mode, Config.PROMPTS[mode])
                    print(f"Logic Test ({mock_text}): {processed}")
                    
            except Exception as e:
                print(f"Error: {e}")
        else:
            print(f"Image not found: {img_path}")

if __name__ == "__main__":
    verify_app()

import os
from model import DARA, Config

def main():
    # Initialize Model
    print("Initializing DARA...")
    dara = DARA()
    
    # Define test cases: image filename -> mode
    test_cases = {
        "food table.jpg": Config.MODE_SCENE,
        "medic.jpg": Config.MODE_MEDICINE,
        "park signs.jpg": Config.MODE_TEXT,
        "sad person.jpg": Config.MODE_EMOTION
    }
    
    base_path = "sampleimages"
    
    print("\nStarting Inference Tests...\n")
    
    for filename, mode in test_cases.items():
        image_path = os.path.join(base_path, filename)
        
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            continue
            
        print(f"--- Testing {filename} in {mode} mode ---")
        try:
            result = dara.detect(image_path, mode)
            print(f"Result: {result['result']}")
            print(f"Audio saved to: {result['audio']}")
        except Exception:
            import traceback
            traceback.print_exc()
        print("\n")

if __name__ == "__main__":
    main()

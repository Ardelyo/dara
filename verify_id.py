from model import DARA, Config
import os

def test_indonesian_support():
    print("🇮🇩 Testing Indonesian Support...")
    
    dara = DARA()
    
    # Test image (use one from sampleimages if available, else use a placeholder or skip)
    image_path = "sampleimages/food table.jpg"
    if not os.path.exists(image_path):
        print(f"⚠️ Warning: {image_path} not found. Skipping inference test.")
        return

    print(f"Testing with {image_path}...")
    
    # Test Scene Mode in Indonesian
    print("\n--- Testing Scene Mode (ID) ---")
    result = dara.detect(image_path, mode=Config.MODE_SCENE, language="id")
    print(f"Result: {result['result']}")
    print(f"Audio: {result['audio']}")
    
    # Test Emotion Mode in Indonesian (using a different image if possible, but same logic applies)
    print("\n--- Testing Emotion Mode (ID) ---")
    # Just reusing the same image to test the translation logic, even if the content isn't perfect for emotion
    result = dara.detect(image_path, mode=Config.MODE_EMOTION, language="id")
    print(f"Result: {result['result']}")
    
    print("\n✅ Verification Complete!")

if __name__ == "__main__":
    test_indonesian_support()

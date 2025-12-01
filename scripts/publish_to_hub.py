from huggingface_hub import HfApi, create_repo, upload_folder
import os
import shutil

def publish_model():
    print("🚀 Preparing to publish DARA to Hugging Face Hub...")
    
    # Configuration
    model_name = "dara-v1" # Change this to your desired model name
    username = input("Enter your Hugging Face username: ")
    repo_id = f"{username}/{model_name}"
    local_model_path = "./dara_model_final"
    
    # Check if model exists
    if not os.path.exists(local_model_path):
        print(f"❌ Error: Model path '{local_model_path}' not found. Please run train.py first.")
        return

    # Login check
    print("\nChecking authentication...")
    os.system("huggingface-cli login")
    
    try:
        # Create Repo
        print(f"\nCreating repository: {repo_id}")
        create_repo(repo_id, repo_type="model", exist_ok=True)
        
        # Upload
        print(f"\nUploading model files from {local_model_path}...")
        api = HfApi()
        api.upload_folder(
            folder_path=local_model_path,
            repo_id=repo_id,
            repo_type="model",
        )
        
        print(f"\n✅ Successfully published to: https://huggingface.co/{repo_id}")
        print("You can now use this model with DARA!")
        
    except Exception as e:
        print(f"\n❌ Error publishing model: {e}")

if __name__ == "__main__":
    publish_model()

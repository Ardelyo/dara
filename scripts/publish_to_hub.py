import os
import argparse
from huggingface_hub import HfApi, create_repo, login, get_token
from huggingface_hub.utils import LocalTokenNotFoundError

def publish_model():
    print("🚀 Preparing to publish DARA to Hugging Face Hub...")
    
    parser = argparse.ArgumentParser(description="Publish DARA model to Hugging Face Hub")
    parser.add_argument("--username", type=str, help="Hugging Face username")
    parser.add_argument("--token", type=str, help="Hugging Face write token")
    parser.add_argument("--model_name", type=str, default="dara-v1", help="Name of the model repository")
    parser.add_argument("--model_path", type=str, default="./dara_model_final", help="Path to the saved model")
    
    args = parser.parse_args()
    
    # 1. Authentication
    token = args.token
    if not token:
        try:
            token = get_token()
            if not token:
                raise LocalTokenNotFoundError
            print("✅ Found local Hugging Face token.")
        except LocalTokenNotFoundError:
            print("\n⚠️ No Hugging Face token found.")
            print("Please find your token at: https://huggingface.co/settings/tokens")
            token = input("Enter your Hugging Face Write Token: ").strip()
            if not token:
                print("❌ Token is required.")
                return
            login(token=token, add_to_git_credential=True)
    
    # 2. Get Username
    api = HfApi(token=token)
    if not args.username:
        try:
            user_info = api.whoami()
            username = user_info['name']
            print(f"✅ Logged in as: {username}")
        except Exception as e:
            print(f"❌ Error getting user info: {e}")
            return
    else:
        username = args.username

    repo_id = f"{username}/{args.model_name}"
    local_model_path = args.model_path
    
    # 3. Check Model Path
    if not os.path.exists(local_model_path):
        print(f"❌ Error: Model path '{local_model_path}' not found.")
        print("Please run 'python scripts/train.py' first to generate the model.")
        return

    try:
        # 4. Create Repo
        print(f"\nCreating repository: {repo_id}")
        create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
        
        # 5. Upload
        print(f"\nUploading model files from {local_model_path}...")
        api.upload_folder(
            folder_path=local_model_path,
            repo_id=repo_id,
            repo_type="model",
            token=token
        )
        
        print(f"\n✅ Successfully published to: https://huggingface.co/{repo_id}")
        print("\nTo use this model:")
        print(f"1. Update Config.MODEL_ID = '{repo_id}' in src/dara/config.py")
        print("   OR")
        print(f"2. Pass model_id='{repo_id}' when initializing DARA()")
        
    except Exception as e:
        print(f"\n❌ Error publishing model: {e}")

if __name__ == "__main__":
    publish_model()

from model import DARA
from dataset import DARADataset
import torch

def verify():
    print("Initializing DARA for processor...")
    dara = DARA()
    processor = dara.processor
    
    print("Loading Dataset...")
    dataset = DARADataset("data/train", processor)
    
    print(f"Dataset Length: {len(dataset)}")
    
    # Check Vocab Size & Embedding Size
    print(f"Model Vocab Size: {dara.model.config.vocab_size}")
    print(f"Embedding Layer Size: {dara.model.get_input_embeddings().weight.shape}")
    print(f"Decoder Start Token ID: {dara.model.config.decoder_start_token_id}")
    
    if len(dataset) > 0:
        item = dataset[0]
        print("First Item Keys:", item.keys())
        for k, v in item.items():
            if isinstance(v, torch.Tensor):
                print(f"{k} shape: {v.shape}, dtype: {v.dtype}")
                if "ids" in k or "labels" in k:
                    print(f"  {k} min: {v.min()}, max: {v.max()}")
            else:
                print(f"{k}: {v}")
                
        # Try Forward Pass
        print("\nRunning Forward Pass...")
        try:
            # Add batch dimension and move to device
            batch = {k: v.unsqueeze(0).to(dara.device) for k, v in item.items()}
            
            # Ensure input_ids are LongTensor
            if "input_ids" in batch:
                batch["input_ids"] = batch["input_ids"].long()
            if "labels" in batch:
                batch["labels"] = batch["labels"].long()
                
            outputs = dara.model(**batch)
            print("Forward Pass Successful!")
            print("Loss:", outputs.loss)
        except Exception as e:
            print("Forward Pass Failed!")
            print(e)
            import traceback
            traceback.print_exc()
    else:
        print("Dataset is empty!")

if __name__ == "__main__":
    verify()

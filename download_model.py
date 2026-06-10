from transformers import RobertaTokenizer, RobertaModel

def download_roberta():
    print("Downloading/Caching RoBERTa tokenizer...")
    RobertaTokenizer.from_pretrained("roberta-base")
    
    print("Downloading/Caching RoBERTa base model...")
    RobertaModel.from_pretrained("roberta-base")
    
    print("Download complete!")

if __name__ == "__main__":
    download_roberta()

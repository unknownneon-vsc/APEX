from transformers import GPT2LMHeadModel, GPT2Tokenizer

def test_transformers():
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        print("Transformers library imported successfully.")
    except Exception as e:
        print(f"Error importing transformers library: {e}")

if __name__ == "__main__":
    test_transformers()
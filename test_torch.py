import torch

def test_torch():
    try:
        print(f"Torch version: {torch.__version__}")
        print("Torch library imported successfully.")
    except Exception as e:
        print(f"Error importing torch library: {e}")

if __name__ == "__main__":
    test_torch()
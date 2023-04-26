library

The `stablelm_interface` library provides a simple interface for working with the StableLM language model from StabilityAI. It includes functions for initializing the model, generating text, and customizing the text generation process.

## Functions

### initialize_model

```
def initialize_model(drive_folder="/content/gdrive/MyDrive/my_folder", model_name="stabilityai/stablelm-tuned-alpha-7b"):
"""
    Load or save a pre-trained Hugging Face model and tokenizer from/to Google Drive.
    
    Parameters:
    - model_name (str): The name or path of the pre-trained model to use.
    - drive_folder (str): The path to the folder where the model and tokenizer should be saved or loaded from (default: "/content/gdrive/MyDrive/my_folder").
    
    Returns:
    - tokenizer: The Hugging Face tokenizer object.
    - model: The Hugging Face model object.
"""
```

This function loads or saves a pre-trained Hugging Face model and tokenizer from/to Google Drive. If the model is not already saved in Google Drive, it will be loaded and then saved to the specified folder. If the model is already saved, it will be loaded from Google Drive.

### generate_text

```
def generate_text(model, tokenizer, user_prompt, system_prompt="", max_new_tokens=128, temperature=0.7, top_k=0, top_p=0.9, do_sample=True):
    """
    Generate text with a Hugging Face model and a user prompt.
    
    Parameters:
    - model: The Hugging Face model object.
    - tokenizer: The Hugging Face tokenizer object.
    - user_prompt (str): The prompt to use as a starting point for text generation.
    - max_new_tokens (int): The maximum number of new tokens to generate (default: 128).
    - temperature (float): Controls the "creativity" of the generated text (default: 0.7).
    - top_k (int): Controls the "quality" of the generated text by limiting the number of candidate tokens at each step (default: 0).
    - top_p (float): An alternative way to control the "quality" of the generated text by selecting from the smallest possible set of tokens whose cumulative probability exceeds the probability threshold (default: 0.9).
    - do_sample (bool): Whether to use sampling or greedy decoding (default: True).
    
    Returns:
    - completion (str): The generated text.
    """
```

This function generates text from a given prompt using the StableLM language model. It takes a user prompt and an optional system prompt, and generates text using the specified parameters.

# Example Usage
Here is an example of how to use the stablelm_interface library:
```
from stablelm_interface import initialize_model, generate_text

# Initialize the model
tokenizer, model = initialize_model("stabilityai/stablelm-base-alpha-7b")

# Generate text
user_prompt = "Can you write a song about a pirate at sea?"
generated_text = generate_text(model, tokenizer, user_prompt)

# Print the generated text
print(generated_text)
```
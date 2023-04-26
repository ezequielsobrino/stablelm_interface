import os
from google.colab import drive
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
          stop_ids = [50278, 50279, 50277, 1, 0]
          for stop_id in stop_ids:
              if input_ids[0][-1] == stop_id:
                  return True
          return False


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
    # Check if the model is already saved in Google Drive
    if not os.path.exists(drive_folder):
        os.mkdir(drive_folder)

    if not os.path.exists(os.path.join(drive_folder, model_name)):
        # Load the tokenizer and the model
        pbar = tqdm(total=3)
        pbar.set_description(f"Loading {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        pbar.update(1)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        pbar.update(1)

        # Save the model to Google Drive
        pbar.set_description(f"Saving {model_name} to Google Drive")
        model.save_pretrained(os.path.join(drive_folder, model_name))
        pbar.update(1)
        tokenizer.save_pretrained(os.path.join(drive_folder, model_name))
        pbar.close()
    else:
        # If the model is already saved, load it from Google Drive
        pbar = tqdm(total=2)
        pbar.set_description(f"Loading {model_name} from Google Drive")
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(drive_folder, model_name))
        pbar.update(1)
        model = AutoModelForCausalLM.from_pretrained(os.path.join(drive_folder, model_name))
        pbar.update(1)
        pbar.close()

    return tokenizer, model

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

    prompt = ""

    # Process the user prompt
    if system_prompt != "":
        prompt = f"{system_prompt}{user_prompt}"
    else:
        prompt = user_prompt

    # Generate
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs.to(model.device)
    tokens = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=StoppingCriteriaList([StopOnTokens()])
    )

    # Extract out only the completion tokens
    completion_tokens = tokens[0][inputs['input_ids'].size(1):]
    completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)

    return completion

def add_one(number):
    return number + 1
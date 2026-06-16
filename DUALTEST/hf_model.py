
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_MODELS = {}


def load_hf_model(model_name):

    if model_name in _MODELS:
        return _MODELS[model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto"
    )

    model.eval()

    _MODELS[model_name] = (tokenizer, model)

    return tokenizer, model


def generate_completion(
    prefix,
    model_name,
    max_new_tokens=64
):

    tokenizer, model = load_hf_model(model_name)

    inputs = tokenizer(
        prefix,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

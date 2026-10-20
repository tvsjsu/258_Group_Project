import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.prompt import format_prompt


MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"


class FunctionCallingModel:
    def __init__(self, model_name=MODEL_NAME, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()

    def generate(self, query, tools, max_new_tokens=512):
        prompt = format_prompt(query, tools)
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(output[0, inputs.shape[1]:], skip_special_tokens=True)

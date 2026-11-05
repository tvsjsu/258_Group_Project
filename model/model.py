import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.prompt import format_prompt


MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"


class FunctionCallingModel:
    def __init__(self, model_name=MODEL_NAME, device=None, dtype="auto"):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Retain checkpoint precision on CUDA; use float32 by default elsewhere.
        load_dtype = {
            "auto": "auto" if torch.device(device).type == "cuda" else torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=load_dtype,
        ).to(device)
        self.model.eval()

    def generate(self, query, tools, max_new_tokens=512):
        """Generate one response, retaining the original single-example interface."""
        return self.generate_batch([query], [tools], max_new_tokens=max_new_tokens)[0]

    def generate_batch(self, queries, tools, max_new_tokens=512):
        """Generate responses in input order using one padded GPU/CPU batch."""
        if len(queries) != len(tools):
            raise ValueError("queries and tools must have the same length")
        if not queries:
            return []
        prompts = []
        for query, definitions in zip(queries, tools):
            messages = [{"role": "user", "content": format_prompt(query, definitions)}]
            prompts.append(self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            ))
        inputs = self.tokenizer(
            prompts, padding=True, return_tensors="pt", add_special_tokens=False
        ).to(self.model.device)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # All inputs share the padded width, including examples with shorter prompts.
        generated = output[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

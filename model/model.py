import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.prompt import build_messages


MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"


class FunctionCallingModel:
    def __init__(self, model_name=MODEL_NAME, device=None, dtype="auto", adapter_name=None, load_in_4bit=False):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        revision = None
        if adapter_name is not None:
            from peft import PeftConfig
            revision = PeftConfig.from_pretrained(adapter_name).revision
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_name or model_name)
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
        options = {"dtype": load_dtype}
        if revision is not None:
            options["revision"] = revision
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            if torch.device(device).type != "cuda":
                raise ValueError("Four-bit evaluation requires a CUDA device")
            compute = torch.bfloat16 if load_dtype == "auto" and torch.cuda.is_bf16_supported() else load_dtype
            if compute == "auto":
                compute = torch.float32
            options.update(
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute,
                ),
                device_map={"": device},
            )
        self.quantization_settings = options["quantization_config"].to_dict() if load_in_4bit else None
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **options)
        if not load_in_4bit:
            self.model = self.model.to(device)
        if adapter_name is not None:
            from peft import PeftModel
            inference_dtype = self.model.dtype
            self.model = PeftModel.from_pretrained(
                self.model, adapter_name, is_trainable=False, autocast_adapter_dtype=False,
            )
            if not load_in_4bit:
                self.model = self.model.to(dtype=inference_dtype)
        self.adapter_parameter_dtypes = sorted({
            str(parameter.dtype) for name, parameter in self.model.named_parameters() if "lora_" in name
        })
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
            messages = build_messages(query, definitions)
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

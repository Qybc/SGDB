"""Optional adapter for the frozen text-only Qwen3-VL explanation backend.

The supplied CPU reproduction uses recorded responses; it never imports this
module's optional model dependencies or makes a new model call.
"""
from pathlib import Path


class FrozenQwenGenerator:
    """Use with pipeline.explain(report, generator, language).

    Supply an existing local Qwen3-VL-8B-Instruct directory. device_index is the
    logical CUDA index after the caller has selected its permitted physical GPU.
    Historical generation used greedy decoding, 640 output tokens and batches of 2.
    """

    def __init__(self, model_path, device_index=0):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.model_path = Path(model_path)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, local_files_only=True
        )
        self.processor.tokenizer.padding_side = 'left'
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map={'': int(device_index)},
            attn_implementation='sdpa',
        ).eval().requires_grad_(False)

    def generate_batch(self, conversations):
        assert 0 < len(conversations) <= 2
        texts = [
            self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in conversations
        ]
        inputs = self.processor(
            text=texts, padding=True, return_tensors='pt'
        ).to(self.model.device)
        prefix = inputs.input_ids.shape[-1]
        assert prefix < 6500
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=640, do_sample=False, use_cache=True
            )
        assert all(not p.requires_grad for p in self.model.parameters())
        return [
            self.processor.decode(row[prefix:], skip_special_tokens=True)
            for row in generated
        ]

    def __call__(self, messages):
        return self.generate_batch([messages])[0]

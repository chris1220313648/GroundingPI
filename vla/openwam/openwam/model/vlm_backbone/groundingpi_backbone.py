"""GroundingPI native image packing and differentiable layerwise features."""
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
import torch
import torch.nn.functional as F
import torch.distributed as dist
from openwam.model.vlm_backbone.hf_vlm_base import HFVlmBackbone
from openwam.model.vlm_backbone.locate_anything_backbone import LocateAnythingBackbone


class GroundingPIBackbone(HFVlmBackbone):
    _allow_remote_code = True
    EXTRA_TENSOR_KEYS = HFVlmBackbone.EXTRA_TENSOR_KEYS + ("patch_positions",)

    def __init__(self, *args, grounding_generation_prompt=False, **kwargs):
        self.grounding_generation_prompt = grounding_generation_prompt
        super().__init__(*args, **kwargs)

    def load_backbone(self, checkpoint_path, *, dtype, load_pretrained):
        LocateAnythingBackbone._isolate_dynamic_module_cache()
        processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
        processor.tokenizer.padding_side = "right"
        if load_pretrained:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path, trust_remote_code=True, dtype=dtype,
                attn_implementation="sdpa",
            )
        else:
            config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).to(dtype=dtype)
        model.config.use_cache = False
        return model, processor

    def prepare_vlm_inputs(self, prompts, images):
        return self._prepare_inputs(prompts, images)

    def prepare_grounding_inputs(self, prompts, images, answers):
        if not self.grounding_generation_prompt:
            raise ValueError("Grounding supervision requires grounding_generation_prompt=true")
        return self._prepare_inputs(prompts, images, answers)

    def _prepare_inputs(self, prompts, images, answers=None):
        if not prompts or len(prompts) != len(images):
            raise ValueError("Expected one observation image or image list per prompt")
        image_sets = [list(x) if isinstance(x, (list, tuple)) else [x] for x in images]
        if any(len(x) not in (1, 3) or any(im is None for im in x) for x in image_sets):
            raise ValueError("Each observation requires one composite or three separate images")
        pixels = self.processor.image_processor(images=[im for views in image_sets for im in views], return_tensors="pt")
        merge = self.processor.image_processor.merge_size ** 2
        grids = pixels["image_grid_thw"]
        if len(grids) != sum(map(len, image_sets)):
            raise ValueError("Image grids and prompts must align")
        texts, prefix_lengths = [], []
        if answers is not None and len(answers) != len(prompts):
            raise ValueError("Expected one answer (or None) per prompt")
        grid_offset = 0
        for i, (prompt, views) in enumerate(zip(prompts, image_sets)):
            sample_grids = grids[grid_offset:grid_offset + len(views)]
            grid_offset += len(views)
            sizes = [int(grid.prod()) for grid in sample_grids]
            if any(size <= 0 or size % merge for size in sizes):
                raise ValueError("Image grid must be divisible by merge_size squared")
            messages = [{"role": "user", "content": [
                *[{"type": "image", "image": "provided"} for _ in views], {"type": "text", "text": prompt}
            ]}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=getattr(self, "grounding_generation_prompt", False), enable_thinking=False
            )
            token = "<|image_pad|>"
            def expand(text):
                parts = text.split(token)
                if len(parts) != len(views) + 1:
                    raise ValueError("Image placeholders must match observation images")
                return parts[0] + "".join(token * (size // merge) + part for size, part in zip(sizes, parts[1:]))
            prefix = expand(text)
            if answers is not None:
                prefix_ids = self.processor.tokenizer(prefix, add_special_tokens=False)["input_ids"]
                prefix_lengths.append(len(prefix_ids))
                if answers[i] is not None:
                    full = expand(self.processor.apply_chat_template(
                        messages + [{"role": "assistant", "content": answers[i]}],
                        tokenize=False, add_generation_prompt=False, enable_thinking=False,
                    ))
                    if not full.startswith(prefix):
                        raise ValueError("Assistant template does not extend generation prefix")
                    # Supervise the answer and im_end, excluding the template's trailing newline.
                    text = full.rstrip("\n")
                    ids = self.processor.tokenizer(text, add_special_tokens=False)["input_ids"]
                    if ids[:len(prefix_ids)] != prefix_ids:
                        raise ValueError("Tokenizer changed the prompt/answer boundary")
                else:
                    text = prefix
            else:
                text = prefix
            texts.append(text)
        # Never truncate an expanded image placeholder sequence.
        inputs = self.processor.tokenizer(texts, padding=True, return_tensors="pt", add_special_tokens=False)
        if inputs["input_ids"].shape[1] > self._max_length:
            raise ValueError(f"Packed input exceeds max_length={self._max_length}; increase it without truncating images")
        if answers is not None:
            labels = torch.full_like(inputs["input_ids"], -100)
            prefix_mask = torch.zeros_like(inputs["attention_mask"], dtype=torch.bool)
            for i, (length, answer) in enumerate(zip(prefix_lengths, answers)):
                prefix_mask[i, :length] = True
                if answer is not None:
                    valid = inputs["attention_mask"][i].bool()
                    labels[i, length:] = inputs["input_ids"][i, length:]
                    labels[i, ~valid] = -100
            inputs["labels"] = labels
            inputs["action_prefix_mask"] = prefix_mask
        return {**inputs, **pixels}

    def extract_grounding_features(self, vlm_inputs, block_indices):
        """One causal pass; only the observation prefix conditions actions."""
        model_inputs = self._to_model_inputs(vlm_inputs)
        labels = model_inputs.pop("labels")
        prefix_mask = model_inputs.pop("action_prefix_mask")
        outputs = self.run_backbone(model_inputs, output_hidden_states=True)
        if len(outputs.hidden_states) != self.num_layers + 1:
            raise RuntimeError("Unexpected GroundingPI hidden-state layout")
        taps = {i: outputs.hidden_states[i + 1] for i in block_indices}
        valid = labels[:, 1:] != -100
        # Project only supervised positions, rather than all image/prompt tokens.
        selected = outputs.last_hidden_state[:, :-1][valid]
        targets = labels[:, 1:][valid]
        if len(targets):
            # Chunking bounds the vocabulary-sized FP32 logits allocation.
            from torch.utils.checkpoint import checkpoint
            def token_loss(hidden, target):
                return F.cross_entropy(self.vlm_model.lm_head(hidden).float(), target, reduction="sum")
            sums = [checkpoint(token_loss, h, t, use_reentrant=False)
                    for h, t in zip(selected.split(128), targets.split(128))]
            local_sum = torch.stack(sums).sum()
        else:
            # Keep the LM head in the graph even on a rank without labels.
            local_sum = self.vlm_model.lm_head(outputs.last_hidden_state[:, :1]).sum() * 0
        count = valid.sum().to(dtype=torch.float32)
        samples = valid.any(dim=1).sum().to(dtype=torch.float32)
        batch_size = torch.tensor(labels.shape[0], device=labels.device, dtype=torch.float32)
        totals = torch.stack([count, samples, batch_size])
        world = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals)
            world = dist.get_world_size()
        ce = local_sum * world / totals[0].clamp_min(1)
        metrics = {"loss_grounding": ce.detach(), "grounding_tokens": totals[0].detach(),
                   "grounding_sample_fraction": (totals[1] / totals[2].clamp_min(1)).detach()}
        return taps, prefix_mask & model_inputs["attention_mask"].bool(), ce, metrics

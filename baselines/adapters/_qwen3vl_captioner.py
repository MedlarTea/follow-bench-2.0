"""Qwen3-VL image captioner used by TrackVLA to generate appearance-aware
follow prompts.

Given a cropped image patch of the target person, returns a short phrase like
``"wearing a red jacket and blue jeans"`` that the TrackVLA adapter splices
into ``"Follow the person <phrase>."`` as the language instruction.

The captioner is intentionally a **separate module** with no dependency on
TrackVLA's own model — it's a small, reusable helper that future planners can
import independently. It also guards all heavy imports behind ``_HAS_QWEN3VL``
so that ``trackvla_adapter`` can fail soft when the extra checkpoint is not
present yet (e.g. on a fresh clone where only the navigation checkpoint was
rsync'd).
"""
from __future__ import annotations

import os
import re
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    _HAS_QWEN3VL = True
    _IMPORT_ERR: Optional[str] = None
except ImportError as _e:
    _HAS_QWEN3VL = False
    _IMPORT_ERR = str(_e)


# Default caption template. The VLM is instructed to respond starting with
# "wearing", so the adapter can splice the answer straight into the TrackVLA
# follow prompt:  "Follow the person " + phrase + "."
DEFAULT_CAPTION_PROMPT = (
    "Describe the clothing of the person in this image in one short phrase. "
    "Start your answer with the word 'wearing' and keep it under 15 words. "
    "Do not describe the background."
)


def _bbox_crop_with_padding(
    rgb: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
    pad_ratio: float = 0.10,
) -> Optional[np.ndarray]:
    """Crop RGB around ``bbox_xyxy`` with a small padding on each side."""
    if rgb is None or bbox_xyxy is None:
        return None
    u0, v0, u1, v1 = [int(v) for v in bbox_xyxy]
    h, w = rgb.shape[:2]
    bw = max(1, u1 - u0)
    bh = max(1, v1 - v0)
    padx = int(round(bw * pad_ratio))
    pady = int(round(bh * pad_ratio))
    u0p = max(0, u0 - padx)
    v0p = max(0, v0 - pady)
    u1p = min(w - 1, u1 + padx)
    v1p = min(h - 1, v1 + pady)
    if u1p <= u0p or v1p <= v0p:
        return None
    patch = rgb[v0p:v1p + 1, u0p:u1p + 1]
    if patch.size == 0:
        return None
    return patch


def _normalize_phrase(raw: str) -> str:
    """Lowercase + trim + ensure the phrase starts with ``wearing``."""
    txt = (raw or "").strip().strip(".")
    # Strip leading "the person is ", "a person ", etc.
    txt = re.sub(r"^(the person is |a person is |this person is |person is )",
                 "", txt, flags=re.IGNORECASE)
    low = txt.lower()
    if not low.startswith("wearing"):
        # Prepend "wearing" if model forgot.
        txt = "wearing " + txt
    return txt


class Qwen3VLCaptioner:
    """Minimal Qwen3-VL-4B-Instruct wrapper returning one-phrase appearance captions.

    Usage
    -----
        cap = Qwen3VLCaptioner(ckpt_dir='<repo>/data/trackvla/LLM_hf/qwen3-vl-4b-instruct')
        phrase = cap.describe(rgb_patch)  # "wearing a red jacket and blue jeans"
        cap.release()                      # free GPU memory after first-frame use

    Typical lifecycle: instantiate once on the first usable frame, call
    ``describe()``, then call ``release()`` to return the VRAM to TrackVLA.
    """

    def __init__(
        self,
        ckpt_dir: str,
        device: str = "cuda",
        dtype: Optional[str] = None,          # 'float16' | 'bfloat16' | None (auto)
        max_new_tokens: int = 48,
        prompt: str = DEFAULT_CAPTION_PROMPT,
    ) -> None:
        if not _HAS_QWEN3VL:
            raise ImportError(
                "Qwen3VLCaptioner requires transformers + torch + PIL. "
                f"Import error: {_IMPORT_ERR}"
            )
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(
                f"Qwen3-VL ckpt dir not found: {ckpt_dir}. Expected the "
                f"HuggingFace checkpoint layout at "
                f"<repo>/data/trackvla/LLM_hf/qwen3-vl-4b-instruct/."
            )
        self.ckpt_dir = ckpt_dir
        self.prompt = str(prompt)
        self.max_new_tokens = int(max_new_tokens)

        # Resolve device + dtype.
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        if dtype is None:
            # Prefer bf16 on Ampere+, fall back to fp16 on older GPUs, fp32 on CPU.
            if device == "cuda":
                self.torch_dtype = (
                    torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16
                )
            else:
                self.torch_dtype = torch.float32
        else:
            self.torch_dtype = getattr(torch, dtype)

        self.processor = AutoProcessor.from_pretrained(ckpt_dir)
        # ``AutoModelForImageTextToText`` is the generic image-text-to-text
        # class and correctly routes to Qwen3-VL's architecture.
        self.model = AutoModelForImageTextToText.from_pretrained(
            ckpt_dir,
            torch_dtype=self.torch_dtype,
        ).to(self.device).eval()

    # ── Main API ──────────────────────────────────────────────────────────────

    def describe(self, rgb_patch: np.ndarray) -> str:
        """Return a short appearance phrase for the person in ``rgb_patch``.

        Parameters
        ----------
        rgb_patch : np.ndarray, HxWx3 uint8, RGB order.
            Already-cropped image of the target; the VLM resizes internally.

        Returns
        -------
        phrase : str
            e.g. ``"wearing a red jacket and blue jeans"``. Guaranteed to
            start with ``wearing``; no trailing period.
        """
        if rgb_patch is None or rgb_patch.size == 0:
            return "wearing ordinary clothes"

        pil = Image.fromarray(rgb_patch.astype(np.uint8))

        # Qwen3-VL uses the chat template convention (same as Qwen2-VL).
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": self.prompt},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )

        inputs = self.processor(
            text=[text],
            images=[pil],
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        # Strip the echoed prompt tokens from the generated sequence.
        input_len = inputs["input_ids"].shape[1]
        gen_ids = out[:, input_len:]
        raw = self.processor.batch_decode(
            gen_ids, skip_special_tokens=True,
        )[0]
        return _normalize_phrase(raw)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def release(self) -> None:
        """Free GPU memory. Call once the first-frame caption is cached."""
        try:
            del self.model
        except Exception:
            pass
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["Qwen3VLCaptioner", "_bbox_crop_with_padding", "DEFAULT_CAPTION_PROMPT"]

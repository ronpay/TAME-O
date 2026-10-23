import base64
import re
from io import BytesIO

import torch
from loguru import logger
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

from .base_model import BaseModel


def resize_image(image: Image.Image, max_size=640):
    """
    Resize an image while maintaining its aspect ratio, ensuring the longest side is max_size pixels.

    Args:
        image (PIL.Image.Image): The image to resize
        max_size (int): Maximum length of the longest dimension

    Returns:
        PIL.Image.Image: Resized image
    """
    width, height = image.size

    if width >= height:
        scaling_factor = max_size / width
    else:
        scaling_factor = max_size / height

    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def pil_image_to_base64(image, image_format="PNG"):
    """
    Convert a PIL.Image.Image object to a Base64 string.

    Args:
        image (PIL.Image.Image): The image to be converted.
        image_format (str): Format to save the image, e.g., "PNG", "JPEG".

    Returns:
        str: The Base64-encoded string of the image.
    """
    image = resize_image(image)
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    buffer.close()
    return image_base64


class QwenModel(BaseModel):
    def __init__(self, model_id="Qwen/Qwen3.5-9B", params=None):
        super().__init__()
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.thinking_enabled = False
        self._template_supports_thinking_flag = True
        self.instruct_system_prompt = (
            "You are a concise vision-language assistant. "
            "Do not reveal chain-of-thought, analysis, or <think> content. "
            "Answer directly in the exact format requested by the user."
        )
        default_params = {
            "do_sample": False,
            "repetition_penalty": 1.0,
        }
        merged_params = default_params if params is None else {**default_params, **params}
        self.params = self._sanitize_generation_params(merged_params)

        dropped_params = sorted(set(merged_params) - set(self.params))
        if dropped_params:
            logger.warning(f"Dropping unsupported generation params for {model_id}: {dropped_params}")
        logger.info(f"Initialized {model_id} with instruct-only, no-thinking generation")

        self.one_id = self.tokenizer.convert_tokens_to_ids("1")
        self.zero_id = self.tokenizer.convert_tokens_to_ids("0")

    @staticmethod
    def model_list():
        return ["Qwen/Qwen3.5-9B"]

    def _sanitize_generation_params(self, params: dict) -> dict:
        supported_params = set(self.model.generation_config.to_dict())
        sanitized_params = {k: v for k, v in params.items() if k in supported_params}

        if not sanitized_params.get("do_sample", False):
            for key in ("temperature", "top_p", "top_k", "min_p"):
                sanitized_params.pop(key, None)

        return sanitized_params

    @staticmethod
    def _normalize_message_content(content):
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        return content

    def _build_messages(self, prompt: str, images: list[Image.Image]) -> list[dict]:
        content = []
        for img in images:
            content.append({"type": "image", "image": f"data:image/png;base64,{pil_image_to_base64(img)}"})
        content.append({"type": "text", "text": prompt})

        return [
            {"role": "system", "content": self._normalize_message_content(self.instruct_system_prompt)},
            {"role": "user", "content": self._normalize_message_content(content)},
        ]

    def _apply_chat_template(self, messages: list[dict]):
        template_kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }

        if self._template_supports_thinking_flag:
            try:
                return self.processor.apply_chat_template(
                    messages,
                    enable_thinking=self.thinking_enabled,
                    **template_kwargs,
                )
            except TypeError:
                self._template_supports_thinking_flag = False
                logger.warning("Qwen chat template does not support enable_thinking; using system prompt fallback only")

        return self.processor.apply_chat_template(messages, **template_kwargs)

    def _strip_reasoning_output(self, output_text: str) -> str:
        cleaned_text = re.sub(r"<think>.*?</think>\s*", "", output_text, flags=re.DOTALL).strip()
        if "</think>" in cleaned_text:
            cleaned_text = cleaned_text.split("</think>", 1)[1].lstrip()
        if "<think>" in cleaned_text:
            cleaned_text = cleaned_text.split("<think>", 1)[0].rstrip()
        return cleaned_text.strip()

    def chat_text(self, prompt: str, max_tokens: int = 512) -> str:
        return self.chat_multi_img(prompt, [], max_tokens)

    def chat_img(self, prompt: str, image: Image.Image, max_tokens: int = 512) -> str:
        return self.chat_multi_img(prompt, [image], max_tokens)

    def chat_multi_img(self, prompt: str, images: list[Image.Image], max_tokens: int = 512) -> str:
        messages = self._build_messages(prompt, images)

        inputs = self._apply_chat_template(messages)
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens, **self.params)

        generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], output_ids)]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        logger.debug(output_text)

        return self._strip_reasoning_output(output_text[0])

import base64
import os
import time
from io import BytesIO
from typing import Optional

import dotenv
import requests
from loguru import logger
from PIL import Image


class APIModel:
    def __init__(self, model_id: str, params={}):
        """
        Initialize APIModel with model_id and API key.

        Args:
            model_id: Model identifier (e.g., "gpt-4-vision-preview")
        """
        super().__init__()
        dotenv.load_dotenv()
        self.model_id = model_id
        self.api_key = os.getenv("API_KEY")
        if not self.api_key:
            raise ValueError("API_KEY environment variable is not set")

        self.api_base = os.getenv("API_URL")
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        self.params = {} if params is None else params

    @staticmethod
    def model_list():
        return ["Qwen/Qwen2.5-VL-72B-Instruct"]

    def _encode_image(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        image = image.convert("RGB")
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def chat_img(self, prompt: str, image: Image.Image, max_tokens: int = 2048) -> str:
        return self.chat_multi_img(prompt, [image], max_tokens)

    def chat_text(self, prompt: str, max_tokens: int = 2048) -> str:
        return self.chat_multi_img(prompt, [], max_tokens)

    def chat_multi_img(
        self, prompt: str, images: list[Image.Image], max_tokens: int = 2048, temperature: float = 0.8
    ) -> str:
        """
        Generate text response for the given prompt and multiple images using OpenAI API.

        Args:
            prompt: Text prompt to process
            images: List of images to analyze (list of PIL Image objects)
            max_tokens: Maximum number of tokens to generate

        Returns:
            str: Generated text response
        """
        content = [{"type": "text", "text": prompt}]

        for image in images:
            base64_image = self._encode_image(image)
            image_content = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
            content.append(image_content)

        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "reasoning_effort": "high",
        }
        payload.update(self.params)

        max_retries = 8
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.api_base}/chat/completions", headers=self.headers, json=payload
                )

                if response.status_code == 429 and attempt < max_retries - 1:
                    retry_delay = 30 * (2**attempt)
                    logger.warning(
                        f"Rate limit (429) hit. Retrying after {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(retry_delay)
                    continue

                response.raise_for_status()

                result = response.json()
                logger.debug(f"API request {payload['messages'][0]['content'][0]}\n Response: {result}")
                return result["choices"][0]["message"]["content"].strip()

            except requests.exceptions.RequestException as e:
                time.sleep(30)
                if attempt == max_retries - 1:
                    raise Exception(f"API request failed after {max_retries} attempts: {str(e)}")

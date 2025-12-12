"""Taken from monorepo (v0.2.1)/LLM.py"""

import os
from abc import ABC, abstractmethod
import base64
from io import BytesIO
import openai
from attrs import define, field
from typing import Optional
import numpy as np
from PIL import Image
from vlfm.utils.prompts import SINGLE_MODEL_GOTO_PROMPT

# Doesn't really matter here
_SAFEGUARD_N_TOKENS = 200_000_000

# Using the very-handwavy 4 letters = 1 token
_SAFEGUARD_N_LETTERS = _SAFEGUARD_N_TOKENS * 4

# For images
_SAFEGUARD_IMAGE_RESOLUTION = 1024


def encode_image_b64(image, format):
    im_file = BytesIO()
    image.save(im_file, format=format.upper())
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    return base64.b64encode(im_bytes).decode("utf-8")


class IRemoteLLM(ABC):
    @abstractmethod
    def image_text_chat(self, prompt: str, image, **kwargs):
        pass

    @abstractmethod
    def text_chat(self, prompt: str, **kwargs):
        pass


@define(kw_only=True, auto_attribs=True)
class OpenAILLM(IRemoteLLM):
    model_id: str
    api_key: str = None
    _url: str = field(default="https://api.openai.com/v1")
    _client: openai.OpenAI = None
    _delay: float = field(default=0.1)
    _temperature: float = field(default=1.0)
    _top_p: float = field(default=0.95)

    def __attrs_post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
        if self._client is None:
            self._client = openai.OpenAI(api_key=self.api_key, base_url=self._url)

    def image_text_chat(self, prompt, image):
        # Handle arrays
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            image_format = "png"
        else:
            image_format = image.format

        if image_format is None or image_format == "None":
            raise ValueError("Wrong image format. I got 'None'. Check how you constructed the image.")
        
        # Handle prompts stylistic choices
        if not prompt.lower().startswith("go to") and not prompt.lower().startswith("navigate to") and not prompt.lower().startswith("go towards") and not prompt.lower().startswith("navigate towards"):
            prompt = SINGLE_MODEL_GOTO_PROMPT.format(OBJECT_DESC=prompt.lower())

        # Safety checks
        height, width = image.size
        if height > _SAFEGUARD_IMAGE_RESOLUTION or width > _SAFEGUARD_IMAGE_RESOLUTION:
            raise ValueError(
                f"Image size safeguard: passed an image of resolution {width} x {height}, larger than the safeguard {_SAFEGUARD_IMAGE_RESOLUTION} x {_SAFEGUARD_IMAGE_RESOLUTION}"
            )
        if len(prompt) > _SAFEGUARD_N_LETTERS:
            print(
                f"!! Warning !! The passed prompt has length {len(prompt)}, greater than the maximum allowed: {_SAFEGUARD_N_LETTERS}. It will be truncated accordingly."
            )
        
        image_bytes = encode_image_b64(image, image_format)
        completion = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt[:_SAFEGUARD_N_LETTERS],
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{image_format.lower()};base64,{image_bytes}"},
                        },
                    ],
                }
            ],
            stream=True,
        )
        stringbuilder = ""

        for chunk in completion:
            token = chunk.choices[0].delta.content
            if token:
                stringbuilder += f"{token}"

        return stringbuilder

    def text_chat(self, prompt):
        if len(prompt) > _SAFEGUARD_N_LETTERS:
            print(
                f"!! Warning !! The passed prompt has length {len(prompt)}, greater than the maximum allowed: {_SAFEGUARD_N_LETTERS}. It will be truncated accordingly."
            )

        completion = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt[:_SAFEGUARD_N_LETTERS],
                        },
                    ],
                }
            ],
            stream=True,
        )
        stringbuilder = ""

        for chunk in completion:
            token = chunk.choices[0].delta.content
            if token:
                stringbuilder += f"{token}"

        return stringbuilder


@define(kw_only=True, auto_attribs=True)
class CerebrasLLM(OpenAILLM):
    api_key: str = None
    _url: str = field(default="https://api.cerebras.ai/v1")

    def __attrs_post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("CEREBRAS_API_KEY")
        if self._client is None:
            self._client = openai.OpenAI(api_key=self.api_key, base_url=self._url)

    def image_text_chat(self, prompt, image):
        raise NotImplementedError(
            "Cerebras still doesn't support multimodal serving. Try another class like `OpenAILLM' or `GeminiLLM'"
        )


@define(kw_only=True, auto_attribs=True)
class VllmLLM(OpenAILLM):
    api_key: str = "EMPTY"
    _port: int = field(default=8000)
    _url: Optional[str] = None

    def __attrs_post_init__(self):
        if self._url is None:
            self._url = f"http://localhost:{self._port}/v1"
        if self._client is None:
            self._client = openai.OpenAI(api_key=self.api_key, base_url=self._url)

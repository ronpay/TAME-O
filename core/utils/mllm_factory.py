from typing import Dict, Optional

from .models.api_model import APIModel
from .models.qwen_model import QwenModel


def MLLMFactory(model_id: str, params: Optional[Dict] = None):
    model_classes = [
        QwenModel,
        APIModel,
    ]
    for model_class in model_classes:
        if model_id in model_class.model_list():
            return model_class(model_id, params)
    raise ValueError(f"Model {model_id} not supported")

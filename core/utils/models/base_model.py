from PIL import Image


class BaseModel:
    def __init__(self):
        pass

    @staticmethod
    def model_list():
        raise NotImplementedError("Model list not implemented")

    def chat(self, text: str, image: Image.Image):
        raise NotImplementedError("Chat method not implemented")

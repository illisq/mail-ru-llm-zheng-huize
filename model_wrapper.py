import traceback
import pickle
import gpt_lm
import stat_lm
import llama_lm
import os

class ModelWrapper:
    def __init__(self):
        self.model = None
        self.current_model_name = None
        self.generate_kwargs = None
        self.result = ""
    def load(self, model_name: str, test_inference: bool = True) -> (bool, str):
        try:
            if model_name == 'StatLM':#n-grams
                self.model, self.generate_kwargs = stat_lm.construct_model()
            elif model_name == 'GPT':
                self.model, self.generate_kwargs = gpt_lm.construct_model()
            # elif model_name == 'Llama': #if your computer can't hold this model ,you can delet this
            #     self.model, self.generate_kwargs = llama_lm.construct_model()
            else:
                return False, f"Модель {model_name} еще не поддерживается"
        except Exception as e:
            print("TRACEBACK")
            print(traceback.format_exc())
            print("*" * 20)
            return False, f"Error while loading model {model_name}: {e}"
        if test_inference:
            try:
                result = self.model.generate("test", **self.generate_kwargs)
            except Exception as e:
                return False, f"Error while test inference model: {e}"

            if not isinstance(result, str):
                return False, f"Test inference result is not string: {type(result)}"

        self.current_model_name = model_name
        return True, ""


    def generate(self, input_text: str) -> (bool, str):
        if self.model is None or self.current_model_name is None:
            return False, "Need to load model"
        if not isinstance(input_text, str):
            return f"Inputs is not text: {type(input_text)}"
        result = self.model.generate(input_text, **self.generate_kwargs)
        if not isinstance(result, str):
            return False, f"Inference result is not string: {type(result)}"
        return True, result


import transformers
import torch
class GPTWrapper:

    def __init__(self):
        self.model = transformers.GPT2LMHeadModel.from_pretrained("./models/gpt")
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained("./models/gpt")

    def generate(self, input_text, **generation_kwargs):
        inputs = self.tokenizer(input_text, return_tensors='pt')
        inputs.update(generation_kwargs)
        generated_tokens = self.model.generate(**inputs)
        return self.tokenizer.decode(generated_tokens[0])


def construct_model():
    generation_kwargs = {
        "max_new_tokens": 40,
        "num_beams": 2,
        "early_stopping": True,
        "no_repeat_ngram_size": 2
    }
    model = GPTWrapper()
    #print('gpt')
    #print(model)
    # #<gpt_lm.GPTWrapper object at 0x000002583F77C940>
#{'max_new_tokens': 40, 'num_beams': 2, 'early_stopping': True, 'no_repeat_ngram_size': 2}

    print(generation_kwargs)
    return model, generation_kwargs
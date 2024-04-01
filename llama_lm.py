import transformers

class LlamaWrapper:

    def __init__(self):
        self.model = transformers.LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")


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
    model = LlamaWrapper()
    return model, generation_kwargs

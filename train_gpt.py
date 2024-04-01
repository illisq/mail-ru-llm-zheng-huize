from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 准备数据集
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="./data/text.txt",
    block_size=128
)
print('train dataset:',len(train_dataset))

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)


training_args = TrainingArguments(
    output_dir="./models/gpt",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    save_steps=10000,
    save_total_limit=2,
)

# begin train
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)
trainer.train()

# save model
model.save_pretrained("./models/gpt")
tokenizer.save_pretrained("./models/gpt")


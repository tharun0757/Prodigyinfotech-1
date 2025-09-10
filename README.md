!pip install transformers datasets accelerate
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset
import os
custom_text = """
In a distant land wrapped in eternal mist, there lived a tribe known as the Whisperkeepers.
Kael was not supposed to be the scribe. He was a stable boy, silent and unremarkable.
But fate had a different plan.
As he traveled through forgotten ruins and across emerald valleys, he met those who feared the scroll.
One evening, under the silver light of twin moons, Kael sat with the scroll.
He wrote not of war, but of peace and unity.
His tale became legend, passed down in stories told by firesides.
The scroll now rests in the Hall of Memory.
"""
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset

# Step 1: Save the text into a file
custom_text = """
In a distant land wrapped in eternal mist, there lived a tribe known as the Whisperkeepers.
Kael was not supposed to be the scribe. He was a stable boy, silent and unremarkable.
But fate had a different plan.
As he traveled through forgotten ruins and across emerald valleys, he met those who feared the scroll.
One evening, under the silver light of twin moons, Kael sat with the scroll.
He wrote not of war, but of peace and unity.
His tale became legend, passed down in stories told by firesides.
The scroll now rests in the Hall of Memory.
"""

# Create the directory and file
os.makedirs("dataset", exist_ok=True)
with open("dataset/custom.txt", "w") as f:
    f.write(custom_text)

# Step 2: Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Step 3: Load dataset (this was failing before because the file didn't exist)
def load_dataset(file_path, tokenizer, block_size=32):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

train_dataset = load_dataset("dataset/custom.txt", tokenizer)

# Step 4: Confirm it's loaded
print("Loaded dataset with", len(train_dataset), "examples.")
from transformers import TrainingArguments
import os
os.environ["WANDB_DISABLED"] = "true"
import os
os.environ["WANDB_DISABLED"] = "true"  # Corrected wandb key

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Use data collator for causal language modeling (GPT-2 is not masked LM)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,                      # Corrected syntax
    per_device_train_batch_size=1,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True
)

# Create the Trainer instance
trainer = Trainer(
    model=model,                             # Fixed typo from modelmmodel
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Start training
trainer.train()
from transformers import pipeline, set_seed

# Initialize generator pipeline
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Set random seed for reproducibility
set_seed(42)

# Custom prompt
prompt = "In the heart of the enchanted forest, "

# Generate text
output = generator(prompt, max_length=150, num_return_sequences=1)

# Print result
print(output[0]['generated_text'])
model.save_pretrained("gpt2-finetuned")
tokenizer.save_pretrained("gpt2-finetuned")
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer from the local directory (adjust path if needed)
model_path = "gpt2-finetuned"
model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)
tokenizer = GPT2Tokenizer.from_pretrained(model_path, local_files_only=True)

# Set up text generation pipeline
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Define the prompt
prompt = "In the heart of the Whispering Hollow"

# Generate text
generated = generator(
    prompt,
    max_length=500,
    num_return_sequences=1,
    do_sample=True,
    temperature=0.9,
    top_k=50,
    top_p=0.95
)

# Save to file
with open("generated_story.txt", "w") as f:
    f.write(generated[0]['generated_text'])

print("Story saved to 'generated_story.txt'")
prompt = "The scroll began to glow in Kael's hands, "
output = generator(prompt, max_length=150, num_return_sequences=1)
print(output[0]['generated_text'])
prompt = "The ancient scroll whispered secrets to Kael, "
output = generator(prompt, max_length=150, num_return_sequences=3)

for i, result in enumerate(output):
    print(f"\n--- Version {i+1} ---\n")
    print(result['generated_text'])
prompt = "The scroll glowed with ancient power."
output = generator(prompt, max_length=150, num_return_sequences=3)

for i, result in enumerate(output):
    print(f"-- Story {i+1} --")
    print(result['generated_text'])
    print()

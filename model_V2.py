# not currently production code
# need to split files out and create config methods
# create training and generation files

import os
import json
import torch
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments, LEDTokenizer, LEDForConditionalGeneration, LEDConfig
from torch.utils.data import Dataset, random_split
from accelerate import DataLoaderConfiguration
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Factorio Blueprint Generation')
    parser.add_argument('--device', default='cpu', help='Device to use for training and inference (cuda or cpu)')
    parser.add_argument('--max_length', type=int, default=16384, help='Maximum sequence length')
    parser.add_argument('--padding', default='max_length', help='Padding strategy')
    parser.add_argument('--truncation', default='longest_first', help='Truncation strategy')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='Number of gradient accumulation steps')
    parser.add_argument('--eval_steps', type=int, default=10, help='Eval Steps')
    parser.add_argument('--save_steps', type=int, default=10, help='Save Steps')
    parser.add_argument('--warmup_steps', type=int, default=10, help='Warmup Steps')
    parser.add_argument('--num_return_sequences', type=int, default=1, help='Number of return sequences')
    parser.add_argument('--encoder_layers', type=int, default=1, help='Number of encoder layers')
    parser.add_argument('--decoder_layers', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--encoder_attention_heads', type=int, default=1, help='Number of attention heads in the encoder')
    parser.add_argument('--decoder_attention_heads', type=int, default=1, help='Number of attention heads in the decoder')
    parser.add_argument('--encoder_ffn_dim', type=int, default=16384, help='Dimensionality of the encoder feed-forward layer')
    parser.add_argument('--decoder_ffn_dim', type=int, default=16384, help='Dimensionality of the decoder feed-forward layer')
    parser.add_argument('--max_encoder_position_embeddings', type=int, default=16384, help='Dimensionality of the encoder feed-forward layer')
    parser.add_argument('--max_decoder_position_embeddings', type=int, default=16384, help='Dimensionality of the decoder feed-forward layer')
    return parser.parse_args()

def load_data(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return [], []

    # Get a list of all JSON files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

    prompts = []
    answers = []

    for file in files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'r') as f:
            data = json.load(f)

            for item in data:
                prompt = item.get("Prompt:", "").strip()
                answer = item.get("Expected Answer:", "")

                if prompt and answer:
                    prompts.append(prompt)
                    answers.append(answer)

    return prompts, answers

class BlueprintDataset(Dataset):
    def __init__(self, encoded_blueprints):
        self.input_ids = encoded_blueprints['input_ids']
        self.attention_mask = encoded_blueprints['attention_mask']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx]}

def preprocess_data(prompts, answers, tokenizer, max_length):
    # Concatenate prompts and answers into a single list of strings
    blueprints = [f"{prompt}\n{answer}" for prompt, answer in zip(prompts, answers)]
    
    # Tokenize the blueprints
    encoded_blueprints = tokenizer.batch_encode_plus(blueprints, add_special_tokens=True, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
    
    # Create a custom dataset
    dataset = BlueprintDataset(encoded_blueprints)
    
    # Split the dataset into train and test sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    return train_dataset, test_dataset

def create_model(model_name, encoder_layers, decoder_layers, encoder_attention_heads, decoder_attention_heads, encoder_ffn_dim, decoder_ffn_dim, max_encoder_position_embeddings, max_decoder_position_embeddings, device):
    config = LEDConfig.from_pretrained(model_name)
    config.encoder_layers = encoder_layers
    config.decoder_layers = decoder_layers
    config.encoder_attention_heads = encoder_attention_heads
    config.decoder_attention_heads = decoder_attention_heads
    config.encoder_ffn_dim = encoder_ffn_dim
    config.decoder_ffn_dim = decoder_ffn_dim
    config.max_encoder_position_embeddings = max_encoder_position_embeddings
    config.max_decoder_position_embeddings = max_decoder_position_embeddings
    model = LEDForConditionalGeneration(config)
    model.to(device)
    return model

def train_model(model, tokenizer, train_dataset, test_dataset, output_dir, epochs, batch_size):
    args = parse_args()
    epochs = args.epochs
    eval_steps = args.eval_steps
    save_steps = args.save_steps
    warmup_steps = args.warmup_steps
    gradient_accumulation_steps = args.gradient_accumulation_steps

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_steps=eval_steps,
        save_steps=save_steps,
        warmup_steps=warmup_steps,
        prediction_loss_only=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()

def generate_blueprint(model, tokenizer, prompt, max_length, device):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)
    global_attention_mask = torch.zeros_like(attention_mask)
    global_attention_mask[:, 0] = 1  # Set global attention on the first token
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        global_attention_mask=global_attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_blueprint = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_blueprint

def main():
    args = parse_args()
    device = args.device
    max_length = args.max_length
    epochs = args.epochs
    batch_size = args.batch_size
    encoder_layers = args.encoder_layers
    decoder_layers = args.decoder_layers
    encoder_attention_heads = args.encoder_attention_heads
    decoder_attention_heads = args.decoder_attention_heads
    encoder_ffn_dim = args.encoder_ffn_dim
    decoder_ffn_dim = args.decoder_ffn_dim
    max_encoder_position_embeddings = args.max_encoder_position_embeddings
    max_decoder_position_embeddings = args.max_decoder_position_embeddings

    # blueprint folders. Change as needed, currently testing areas
    blueprints_folder = 'Blueprints/Prompts' # Linux
    # blueprints_folder = r"blueprints\Prompts" # Windows Internal Drive
    # blueprints_folder = r"blueprints\Prompts" # Windows External Drive

    model_name = 'allenai/led-large-16384'  # Choose the desired LED model
    tokenizer = LEDTokenizer.from_pretrained(model_name)
    model = create_model(model_name, encoder_layers, decoder_layers, encoder_attention_heads, decoder_attention_heads, encoder_ffn_dim, decoder_ffn_dim, max_encoder_position_embeddings, max_decoder_position_embeddings, device)

    prompts, answers = load_data(blueprints_folder)
    print(f"Loaded {len(prompts)} prompts and {len(answers)} answers from {blueprints_folder}")

    train_dataset, test_dataset = preprocess_data(prompts, answers, tokenizer, max_length)
    print(f"Preprocessed {len(train_dataset)} training samples and {len(test_dataset)} test samples")

    output_dir = 'output'
    epochs = epochs
    batch_size = batch_size
    train_model(model, tokenizer, train_dataset, test_dataset, output_dir, epochs, batch_size)

    generated_blueprints = []
    for _ in range(1):
        prompt = '{\n    "blueprint": {'  # Use the {prompt} as the input
        generated_blueprint = generate_blueprint(model, tokenizer, prompt, max_length, device)
        generated_blueprints.append(generated_blueprint)

    print(generated_blueprints)

if __name__ == '__main__':
    main()
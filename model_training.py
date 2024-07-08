from transformers import MistralForCausalLM, MistralConfig
from torch.utils.data import IterableDataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from transformers import PhiForCausalLM, PhiConfig
from transformers import DataCollatorWithPadding
from transformers import TrainerCallback
from transformers import EvalPrediction
from transformers import AutoTokenizer
from torch.nn import CrossEntropyLoss
from collections import defaultdict
from accelerate import Accelerator
from datasets import load_dataset
from datasets import DatasetDict
from datetime import datetime
from datasets import Dataset
from tqdm.auto import tqdm
import numpy as np
import random
import pickle
import shutil
import wandb
import torch
import yaml
import os

# Set CUDA_VISIBLE_DEVICES to only see GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 


print("Loading the tokenizer")
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("En_De_Fr_Py_Tokenizer")
tokenizer.pad_token = tokenizer.eos_token
# Initialize the Accelerator
accelerator = Accelerator()
device = accelerator.device
print(f"Device is {device}")
print("Loading the data")

max_position_embeddings = 1024
language_mapping = {0:'en', 1:'fr', 2:'de', 3:'py'}


with open('tokenized_datasets/tokenized_train_datasets_1kContext_sep4lang.pkl', 'rb') as file:
    shuffle_seed = random.randint(1,999999)
    np.random.seed(shuffle_seed)
    tokenized_train_datasets = pickle.load(file)
    # Shuffle
    tokenized_train_datasets = {key: dataset.shuffle() for key, dataset in tokenized_train_datasets.items()}

# Load the tokenized validation datasets
with open('tokenized_datasets/tokenized_valid_datasets_1kContext_sep4lang.pkl', 'rb') as file:
    tokenized_valid_datasets = pickle.load(file)
    tokenized_valid_datasets = {language_mapping[k]: v for k, v in tokenized_valid_datasets.items()}

used_model = 'distil-gpt2'
random_batches = False
vocab_size=32000
print("Configuring the model")
if 'mistral' in used_model:
    # Provided custom configuration for Mistral
    custom_config = MistralConfig( 
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=1024*4,
        max_position_embeddings=max_position_embeddings,
        model_type="mistral",
        num_attention_heads=16,
        num_hidden_layers=24, 
        num_key_value_heads=8,
        rms_norm_eps=1e-05,
        rope_theta=1000000.0,
        sliding_window=None,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
        transformers_version="4.36.0",
        use_cache=True,
        vocab_size=vocab_size
    )

    print("Initializing the model")
    # Initialize the model with the custom configuration
    model = MistralForCausalLM(config=custom_config)

elif used_model=='phi':
    custom_config = PhiConfig(
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        hidden_size=2048,
        initializer_range=0.02,
        intermediate_size=8192,
        max_position_embeddings=max_position_embeddings,
        model_type="phi",
        num_attention_heads=32,
        num_hidden_layers=24,
        num_key_value_heads=4,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
        transformers_version="4.36.0",
        use_cache=True,
        vocab_size=vocab_size
    )
    
    print("Initializing the model")
    # Initialize the model with the custom configuration
    model = PhiForCausalLM(config=custom_config)

elif used_model == 'gpt2'  :
    custom_config = GPT2Config(
        bos_token_id=1,
        eos_token_id=2,
        vocab_size=vocab_size,
        n_positions=max_position_embeddings,
        n_ctx=1024,
        n_embd=1280,
        n_head=20,
        n_layer=36,
        n_special=0,
    )
    print("Initializing GPT-2 model from scratch")
    model = GPT2LMHeadModel(config=custom_config)

elif used_model == 'distil-gpt2' :
    custom_config = GPT2Config(
        bos_token_id=1,
        eos_token_id=2,
        vocab_size=vocab_size,
        n_positions=max_position_embeddings,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=6,
        n_special=0,
        torch_dtype="bfloat16",
    )
    print("Initializing GPT-2 model from scratch")
    model = GPT2LMHeadModel(config=custom_config)

model_size = f"{round(model.num_parameters()/1_000_000_000,2)}B"
print(f"Number of Parameters = {model_size}")

class CustomCheckpointCallback(TrainerCallback):
    def __init__(self, save_path, metric_for_best_model="loss", greater_is_better=False, trainer=None):
        self.best_metric = None
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        self.save_path = save_path
        self.best_checkpoint_path = os.path.join(save_path, "best_checkpoint")
        self.last_checkpoint_path = os.path.join(save_path, "last_checkpoint")
        self.trainer = trainer
        
        # Ensure the save path exists
        os.makedirs(self.save_path, exist_ok=True)
    
    def on_train_end(self, args, state, control, **kwargs):
        # Logic to overwrite/save the "last" checkpoint
        self._save_checkpoint(state, self.last_checkpoint_path, None)


    def _save_checkpoint(self, state, checkpoint_path, metrics):
        # Ensure the target checkpoint directory exists, create if it doesn't
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save the model and tokenizer using the Trainer's save_model method
        if self.trainer is not None:
            self.trainer.save_model(checkpoint_path)
            if self.trainer.tokenizer is not None:
                self.trainer.tokenizer.save_pretrained(checkpoint_path)
        
        # Save additional information in checkpoint_info.txt
        checkpoint_info_path = os.path.join(checkpoint_path, "checkpoint_info.txt")
        with open(checkpoint_info_path, 'w') as f:
            f.write(f"Epoch: {self.trainer.state.epoch}\n")
            f.write(f"Step: {self.trainer.state.global_step}\n")
            if metrics is not None:
                for key, value in metrics.items():
                    f.write(f"{key}: {value}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

with open('ds_config.yml', 'r') as file:
    ds_config = yaml.load(file, Loader=yaml.FullLoader)

GA_steps = ds_config['deepspeed_config']['gradient_accumulation_steps']
output_dir = f"./{custom_config.model_type}_{model_size}_exp_results_{datetime.now().strftime('%Y-%m-%d %H:%M')}"
epochs=3
train_batch_size = 32
eval_batch_size = 64
len_train_data = sum([x.num_rows for x in tokenized_train_datasets.values()])
num_GPUs = ds_config['num_processes']
num_steps = ((len_train_data*epochs)/(train_batch_size*GA_steps))/num_GPUs
eval_steps = num_steps//16
training_args = TrainingArguments( 
    output_dir=output_dir,                              # Directory for saving outputs
    num_train_epochs=epochs,                            # Number of training epochs
    per_device_train_batch_size=train_batch_size,       # Batch size per device during training
    per_device_eval_batch_size=eval_batch_size,         # Batch size for evaluation
    gradient_accumulation_steps=GA_steps,               # Number of updates steps to accumulate before performing a backward/update pass.
    warmup_steps=0.05*num_steps,                        # Number of warmup steps for learning rate scheduler # Change - To do - ToDo
    weight_decay=0.05,                                  # Weight decay if we apply some.
    logging_steps=1,                                    # Log every X updates steps.
    save_steps=eval_steps,                              # Save checkpoint every eval_steps steps 
    save_total_limit=2,                                 # Keep only the most recent checkpoints to save space.
    eval_steps=eval_steps,                              # Evaluate every eval_steps steps
    max_steps=num_steps,
    evaluation_strategy="steps",                        # Perform evaluation at each specified step
    load_best_model_at_end=True,
    metric_for_best_model="eval_en_loss",
    greater_is_better=False,
    remove_unused_columns=False,
    bf16=True,                                          # Enable bf16 training # Change - To do - ToDo to bf16
    half_precision_backend="deepspeed",                 # Specify DeepSpeed as the backend for FP16
    report_to="wandb",                                  # Report the logs to Weights & Biases (wandb)
    logging_dir='./logs',                               # Directory for storing logs
)

# wandb.init(project="modular_students_pretraining", entity="modular_students", config=training_args.to_dict())
wandb.init(project="modular_student_kd", entity="modular_students", config=training_args.to_dict())
# modular_students
# Update wandb config with our custom configuration
wandb.config.update({"used_architecture": custom_config.model_type})
wandb.config.update({"Model name": used_model})
wandb.config.update({"Full training batch size (per_device_train_batch_size*num_GPUs*gradient_accumulation_steps)": train_batch_size*num_GPUs*GA_steps})
wandb.config.update(custom_config.__dict__)
# Update wandb config with the deepSpeed config
wandb.config.update({"deep_speed_config": ds_config})
wandb.config.update({"num_parameters": f"{round(model.num_parameters()/1_000_000_000,2)}B"})
wandb.config.update({"Number of training samples": len_train_data})
wandb.config.update({"Balanced batches (True/False)": not(random_batches)})
wandb.config.update({"Number of training steps": num_steps})
wandb.config.update({"Number of validation samples": len(tokenized_valid_datasets)})
wandb.config.update({"Train dataset shuffle random seed": shuffle_seed})
wandb.save('model_training.py')
wandb.save('ds_config.yml')

class BalancedLanguageBatchDataset(IterableDataset):
    def __init__(self, datasets, batch_size, device, random_batches=False):
        self.datasets = datasets
        self.device = device
        self.batch_size = batch_size
        self.batch_size_per_lang = batch_size // 4
        self.random_batches = random_batches
        self.iterators = {lang: iter(ds) for lang, ds in datasets.items()}
        self.dataset_sizes = {lang: len(ds) for lang, ds in datasets.items()}
        self.max_dataset_size = max(self.dataset_sizes.values())
        self.total_iterations_needed = self.max_dataset_size // self.batch_size_per_lang  # Total iterations needed to cover the largest dataset
        
    def restart_iterator(self, lang):
        """Restart an iterator for a specific language."""
        self.iterators[lang] = iter(self.datasets[lang])

    def __iter__(self):
        self.current_iteration = 0
        return self

    def __next__(self):
        if self.current_iteration >= self.total_iterations_needed:
            raise StopIteration
        batch = defaultdict(list)

        if self.random_batches:
            for _ in range(self.batch_size):
                lang = random.choice(list(self.datasets.keys()))
                try:
                    item = next(self.iterators[lang])
                except StopIteration:
                    self.restart_iterator(lang)
                    item = next(self.iterators[lang])
                
                for key in item:
                    batch[key].append(item[key])
        
        else:
            for lang in self.iterators.keys():
                for _ in range(self.batch_size_per_lang):
                    try:
                        item = next(self.iterators[lang])
                    except StopIteration:
                        self.restart_iterator(lang)
                        item = next(self.iterators[lang])
                    
                    for key in item:
                        batch[key].append(item[key])

        batch = {k: torch.tensor(v, device=self.device) for k, v in batch.items()}
        self.current_iteration += 1
        return batch

def create_balanced_dataloaders(train_datasets_separated, batch_size):
    train_balanced_ds = BalancedLanguageBatchDataset(train_datasets_separated, batch_size, device=device, random_batches=random_batches)

    train_dataloader = DataLoader(train_balanced_ds, batch_size=None)  # batch_size=None because dataset yields batches directly

    return train_dataloader


class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features):
        # Use parent class to collate lists of features into a batch
        batch = super().__call__(features)
        # Handle 'language' field properly
        if "language" in features[0]:
            # Convert 'language' list to a tensor before adding to the batch
            languages = torch.tensor([feature["language"] for feature in features])
            batch["language"] = languages.to(batch["input_ids"].device)
        return batch


data_collator = CustomDataCollatorWithPadding(tokenizer=tokenizer)


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize a dictionary to store the last non-zero loss for each language
        self.last_non_zero_loss = {"en": None, "de": None, "fr": None, "py": None}

        self.train_dataloader = create_balanced_dataloaders(
            self.train_dataset, 
            batch_size=self.args.per_device_train_batch_size * torch.cuda.device_count()
        )

    # Override get_train_dataloader to use the custom balanced train dataloader
    def get_train_dataloader(self):
        return self.train_dataloader

    def compute_loss(self, model, inputs, return_outputs=False):
        # Remove 'language' from inputs if it exists
        languages = inputs.pop('language', None)
        
        # Call the superclass's compute_loss method
        return super().compute_loss(model, inputs, return_outputs)

    def compute_lang_losses(self, model, inputs, prefix="", return_outputs=False):
        # Assume languages tensor and labels are available in inputs
        languages = inputs.pop("language")
        labels = inputs.get("labels")
        loss_fct = CrossEntropyLoss()

        # Forward pass (compute all at once)
        outputs = model(**inputs)
        logits = outputs.get('logits')

        # Assuming 'en' -> 0, 'fr' -> 1, 'de' -> 2, 'py' -> 3
        language_codes = [0, 1, 2, 3]
        language_labels = ["en", "de", "fr", "py"]

        # Initialize tensor to store losses for each language
        losses = torch.zeros(len(language_labels), device=logits.device)
        counts = torch.zeros_like(losses)

        # Initialize dictionaries to store losses for each language
        loss_dict = {}
        
        # Vectorized computation for each language
        for i, lang_code in enumerate(language_codes):
            lang_mask = (languages == lang_code)
            if lang_mask.any():
                lang_logits = logits[lang_mask]
                lang_labels = labels[lang_mask]
                lang_loss = loss_fct(lang_logits.view(-1, vocab_size), lang_labels.view(-1))
                loss_dict[language_labels[i]] = lang_loss
                losses[i] = lang_loss
                counts[i] = lang_mask.sum()

        if prefix=='train_':
            # Logging losses per language
            log_dict = {f"{prefix}{lang}_loss": loss.item() for lang, loss in zip(language_labels, losses)}
            self.log(log_dict)

        else:
            return {"losses": loss_dict, "logits": logits}


class WandbLoggingCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        # Initialize to store losses for aggregation
        self.metrics_by_step = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            step = state.global_step
            if step not in self.metrics_by_step:
                self.metrics_by_step[step] = {}
            
            # Update the metrics for the current step with new logs
            self.metrics_by_step[step].update(logs)

            # Check if we have all needed metrics for the current step
            expected_keys = {'eval_de_loss', 'eval_en_loss', 'eval_fr_loss', 'eval_py_loss'}
            if expected_keys.issubset(self.metrics_by_step[step].keys()):
                # Calculate average loss and perplexity
                total_loss = sum(self.metrics_by_step[step][key] for key in expected_keys)
                avg_loss = total_loss / len(expected_keys)
                avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()

                # Prepare final log metrics
                wandb_metrics = {
                    **{key: self.metrics_by_step[step][key] for key in expected_keys},
                    'eval_avg_loss': avg_loss,
                    'eval_avg_perplexity': avg_perplexity,
                    'step': step
                }
                if "loss" in logs:
                    wandb_metrics["total_loss"] = logs["loss"]
                    total_perplexity = torch.exp(torch.tensor(logs["loss"])).item()
                    wandb_metrics["total_perplexity"] = total_perplexity


                # Log to Wandb
                wandb.log(wandb_metrics)

                # Clear metrics for this step to prevent re-logging
                del self.metrics_by_step[step]

    def log_metrics(self, metrics, step):
        wandb_metrics = {}
        for key, value in metrics.items():
            # This assumes the metrics dictionary contains <lang>_loss entries during evaluation
            if "_loss" in key:
                wandb_metrics[key] = value
            elif key in ["loss", "eval_loss"]:
                # Handles total loss
                wandb_metrics["total_" + key] = value

        if wandb_metrics:  # Check if there's anything to log
            wandb_metrics["step"] = step
            wandb.log(wandb_metrics)

wandb_logging_callback = WandbLoggingCallback()

save_path = f"./{custom_config.model_type}_{model_size}_exp_results/checkpoints"

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_datasets,
    eval_dataset=tokenized_valid_datasets, 
    data_collator=data_collator,
)

# Assign the callback to the trainer
trainer.add_callback(wandb_logging_callback)


print("Start training ... ")
# Use the Accelerator to prepare the trainer
trainer = accelerator.prepare(trainer)


try:
    trainer.train()
except KeyboardInterrupt:
    print("Keyboard interruption detected.")
finally:
    print("Saving the model and training state before exiting...")
    # Specify the directory to save model, tokenizer, and training state
    save_dir = f"./interrupted_{custom_config.model_type}_{model_size}_training_checkpoint_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    # Ensure the output_dir in training arguments points to the new save directory
    trainer.args.output_dir = save_dir

    # Saving model and tokenizer using Trainer's convenient method
    trainer.save_model(save_dir)
    if trainer.tokenizer is not None:
        trainer.tokenizer.save_pretrained(save_dir)
    
    # Saving optimizer and scheduler states
    trainer.save_state()
    
    # Optionally, save other components like the training arguments if needed for resuming
    try:
        training_args_save_path = f"{save_dir}/training_args.bin"
        torch.save(trainer.args, training_args_save_path)
    except:
        print("Failed to save Training Arguments")

    print(f"Model, tokenizer, and training state saved to {save_dir}. Exiting.")

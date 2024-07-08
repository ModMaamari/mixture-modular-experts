from transformers import MistralForCausalLM, MistralConfig
from transformers import AutoConfig, AutoModelForCausalLM
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
import torch.nn.functional as F
from datetime import datetime
from datasets import Dataset
from tqdm.auto import tqdm
import torch.nn as nn
import numpy as np
import random
import pickle
import shutil
import torch
import wandb
import torch
import time
import yaml
import os


# Set CUDA_VISIBLE_DEVICES to only see GPU 0,1
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
code2lang = {0:'en', 1:'fr', 2:'de', 3:'py'}
lang2code = {v:k for k,v in code2lang.items()}

selected_language = 'en'
load_sample=False
drop_language_feature = True

load_sample_prefix = '' if not load_sample else '_sample'
with open(f'tokenized_datasets/tokenized_train_datasets_1kContext_sep4lang{load_sample_prefix}.pkl', 'rb') as file:
    shuffle_seed = random.randint(1,999999)
    np.random.seed(shuffle_seed)
    tokenized_train_datasets = pickle.load(file)
    # Shuffle
    tokenized_train_datasets = {key: dataset.shuffle() for key, dataset in tokenized_train_datasets.items()}
    if drop_language_feature:
        tokenized_train_datasets = {key: dataset.remove_columns('language') for key, dataset in tokenized_train_datasets.items()}

# Load the tokenized validation datasets
with open(f'tokenized_datasets/tokenized_valid_datasets_1kContext_sep4lang{load_sample_prefix}.pkl', 'rb') as file:
    tokenized_valid_datasets = pickle.load(file)
    tokenized_valid_datasets = {code2lang[k]: v for k, v in tokenized_valid_datasets.items()}
    if drop_language_feature:
        tokenized_valid_datasets = {key: dataset.remove_columns('language') for key, dataset in tokenized_valid_datasets.items()}

vocab_size=32000
print("Initializing the model")
# Initialize Teacher Model (Pretrained GPT2-Medium)
checkpoint_path = 'gpt2_0.34B_exp_results_2024-04-27 08:49/checkpoint-2304'
teacher_config = AutoConfig.from_pretrained(checkpoint_path)
teacher_model = AutoModelForCausalLM.from_pretrained(checkpoint_path, config=teacher_config)
teacher_name = checkpoint_path.split('_exp')[0]+' '+checkpoint_path.split('/')[1]

# Initialize Student Model (untrained Distill GPT2 with 32k vocab_size)
student_config = GPT2Config(
    bos_token_id=1,
    eos_token_id=2,
    layer_norm_epsilon=1e-05,
    n_ctx=1024,
    n_embd=768,
    n_head=12, #12
    n_layer=6, #6
    n_positions=1024,
    vocab_size=32000,
    resid_pdrop=0.1,
    torch_dtype="bfloat16",
)

# Initialize the model with the specified configuration
student_model = GPT2LMHeadModel(student_config)

# student_checkpoint_path = 'KD_en_de_fr_67M_exp_results_2024-05-15 01:53/checkpoint-592'
# student_config = AutoConfig.from_pretrained(student_checkpoint_path)
# student_model = AutoModelForCausalLM.from_pretrained(student_checkpoint_path, config=student_config)

teacher_model = teacher_model.to(device)
student_model = student_model.to(device)

def count_parameters_in_millions(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000

student_model_params_millions = count_parameters_in_millions(student_model)
student_name = f"Distill GPT2 {student_model_params_millions:.2f}M"
teacher_model_params_millions = count_parameters_in_millions(teacher_model)

print(f"Number of parameters in teacher model: {teacher_model_params_millions:.2f} million")
print(f"Number of parameters in student model: {student_model_params_millions:.2f} million")
print(f"The student model is {(student_model_params_millions/teacher_model_params_millions)*100:.2f}% of the teacher model")


with open('ds_config.yml', 'r') as file:
    ds_config = yaml.load(file, Loader=yaml.FullLoader)

class KnowledgeDistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, 
                 len_loss_history=10, adaptive_alpha=True,
                 inverse_KL=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = 1-alpha
        self.adaptive_alpha = adaptive_alpha
        self.inverse_KL = inverse_KL
        self.temperature = temperature
        self.len_loss_history = len_loss_history

GA_steps = ds_config['deepspeed_config']['gradient_accumulation_steps']
epochs=3
train_batch_size = 32
eval_batch_size = 64
len_train_data = tokenized_train_datasets[lang2code[selected_language]].num_rows
num_GPUs = ds_config['num_processes']
num_steps = int(((len_train_data*epochs)/(train_batch_size*GA_steps))/num_GPUs)
eval_steps = num_steps//16
len_loss_history = max(20,num_steps//100)
one_loss_at_a_time=False
output_dir = f"./KD_{selected_language}_{int(student_model_params_millions)}M_myKDloss_exp_results_{datetime.now().strftime('%Y-%m-%d %H:%M')}"

student_training_args = KnowledgeDistillationTrainingArguments( 
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
    max_steps=num_steps,                                # Maximum number of training steps
    evaluation_strategy="steps",                        # Perform evaluation at each specified step
    load_best_model_at_end=True,                        # Load the best model found during training at the end of training
    metric_for_best_model=f"eval_{selected_language}_loss",                  # Use eval_loss to determine the best model.
    greater_is_better=False,                            # Best model is the one with the lowest eval_loss
    remove_unused_columns=False,                        # Do not remove columns not required by the model
    bf16=True,                                          # Enable bf16 training # Change
    half_precision_backend="deepspeed",                 # Specify DeepSpeed as the backend for FP16
    report_to="wandb",                                  # Report the logs to Weights & Biases (wandb)
    logging_dir='./logs',                               # Directory for storing logs
    adaptive_alpha=True,                                # Activate adaptive alpha
    inverse_KL=True,                                    # Use inverse KL loss instead of normal KL
    alpha = 0.5,                                        # Distillation parameter for controlling the trade-off in loss function
    len_loss_history=len_loss_history,                  # Length of loss history for adaptive alpha
)

adaptive_alpha_config = {
    "adaptive_alpha": student_training_args.adaptive_alpha,
    "len_loss_history": student_training_args.len_loss_history,
    "alpha": student_training_args.alpha,
    "beta": student_training_args.beta
}
KD_config = {
    "Distilled language": selected_language,
    "temperature": student_training_args.temperature,
    "inverse_KL": student_training_args.inverse_KL
}

# Initialize wandb only in the main process
wandb.init(project="modular_student_kd", entity="modular_students", config=student_training_args.to_dict())
# Update wandb config with our custom configuration
wandb.config.update({"student_model": student_name})
wandb.config.update({"teacher_model": teacher_name})
wandb.config.update({"Full training batch size (per_device_train_batch_size*num_GPUs*gradient_accumulation_steps)": train_batch_size*num_GPUs*GA_steps})
wandb.config.update({'student_config':student_config.__dict__})
wandb.config.update({'teacher_config':teacher_config.__dict__})
wandb.config.update({'student_size (million)':student_model_params_millions})
wandb.config.update({'teacher_size (million)':teacher_model_params_millions})
wandb.config.update({'num_GPUs':num_GPUs})
wandb.config.update({'one_loss_at_a_time':one_loss_at_a_time})
wandb.config.update({"adaptive_alpha_config": adaptive_alpha_config})
wandb.config.update({"KD_config": KD_config})

# Update wandb config with the deepSpeed config
wandb.config.update({"deep_speed_config": ds_config})
wandb.config.update({"Number of training samples": len_train_data})
wandb.config.update({"Number of training steps": num_steps})
wandb.config.update({"Number of validation samples": len(tokenized_valid_datasets)})
wandb.config.update({"Train dataset shuffle random seed": shuffle_seed})
wandb.save('KD_training_one_student.py')
wandb.save('ds_config.yml')


class KnowledgeDistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, one_loss_at_a_time=False, teacher_trainer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_trainer = teacher_trainer
        self.teacher_model.eval()
        self.teacher_loss_is_cached = False
        self.teacher_loss = np.inf
        self.teacher_perplexity = np.inf
        self.loss_ce_history = []
        self.loss_kd_history = []
        self.phase = 'train'
        self.one_loss_at_a_time = one_loss_at_a_time
        self.alternate = False

    
    def _calculate_differential_weights(self, k):
        if len(self.loss_ce_history) < k or len(self.loss_kd_history) < k:
            return self.args.alpha, self.args.beta  # Initial values

        
        # Differential calculation for student_loss_ce and student_loss_kd
        def differential_weight(loss_history):
            diffs = [loss_history[-i] - loss_history[-i-1] for i in range(1, k)]
            avg_loss = sum(loss_history[-k:]) / k
            normalized_diff = sum(diffs) / avg_loss if avg_loss != 0 else 0
            # Convert normalized difference to a weight between 0 and 1
            weight = 1 - min(max(normalized_diff, 0), 1)  # Clamping between 0 and 1
            return weight

        alpha = differential_weight(self.loss_ce_history)
        beta = differential_weight(self.loss_kd_history)

        # Normalize alpha and beta to ensure their sum is 1
        total = alpha + beta
        return alpha / total, beta / total

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     # Compute loss and outputs for the student model using the superclass method
    #     student_loss_ce, student_outputs = super().compute_loss(model, inputs, return_outputs=True)
    #     student_logits = student_outputs.logits

    #     if self.phase == 'train':
    #         with torch.no_grad():
    #             teacher_loss_ce, teacher_outputs = super().compute_loss(self.teacher_model, inputs, return_outputs=True)
    #             teacher_logits = teacher_outputs.logits

    #         # Compute distillation loss, normal KL or inverse KL based on args
    #         if not self.args.inverse_KL:
    #             student_loss_kd = nn.functional.kl_div(
    #                 input=F.log_softmax(student_logits / self.args.temperature, dim=-1).float(),
    #                 target=F.softmax(teacher_logits / self.args.temperature, dim=-1).float(),
    #                 reduction='batchmean'
    #             ) * (self.args.temperature ** 2)
    #         else:
    #             # Inverse KL
    #             teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.bfloat16)
    #             inf_mask = torch.isinf(student_logits)
    #             logprobs = F.log_softmax(student_logits, dim=-1, dtype=torch.bfloat16)
    #             prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
    #             x = torch.sum(prod_probs, dim=-1).view(-1)
    #             loss_mask = (inputs["input_ids"] != tokenizer.pad_token_id).int().view(-1)
    #             student_loss_kd = -torch.sum(x * loss_mask, dim=0) / torch.sum(loss_mask, dim=0)

    #         # Store the current losses
    #         self.loss_ce_history.append(student_loss_ce.item())
    #         self.loss_kd_history.append(student_loss_kd.item())

    #         if self.one_loss_at_a_time:
    #             # Alternate between student_loss_ce and student_loss_kd
    #             if self.alternate:
    #                 student_combined_loss = student_loss_kd
    #             else:
    #                 student_combined_loss = student_loss_ce
    #             self.alternate = not self.alternate
    #         else:
    #             # Calculate differential weights
    #             if self.args.adaptive_alpha:
    #                 k = self.args.len_loss_history
    #                 self.args.alpha, self.args.beta = self._calculate_differential_weights(k)

    #             # Combined loss with differential weights
    #             student_combined_loss = self.args.alpha * student_loss_ce + self.args.beta * student_loss_kd

    #         wandb.log({
    #             f"{self.phase}_loss": student_combined_loss.item(),
    #             f"{self.phase}_loss_ce": student_loss_ce.item(),
    #             f"{self.phase}_loss_kd": student_loss_kd.item(),
    #             f"{self.phase}_alpha": self.args.alpha,
    #             f"{self.phase}_beta": self.args.beta,
    #             f"{self.phase}_teacher_loss": teacher_loss_ce.item(),
    #             f"{self.phase}_student_perplexity": torch.exp(student_loss_ce).item(),
    #             f"{self.phase}_teacher_perplexity": torch.exp(teacher_loss_ce).item()
    #         })

    #     elif self.phase == 'eval':
    #         student_combined_loss = student_loss_ce
    #         wandb.log({
    #             f"{self.phase}_loss": student_loss_ce.item(),
    #             f"{self.phase}_loss_ce": student_loss_ce.item(),
    #             f"{self.phase}_student_perplexity": torch.exp(student_loss_ce).item()
    #         })

    #     return (student_combined_loss, student_outputs) if return_outputs else student_combined_loss

    # After adding the new loss
    def compute_loss(self, model, inputs, return_outputs=False):
        # Compute loss and outputs for the student model using the superclass method
        student_loss_ce, student_outputs = super().compute_loss(model, inputs, return_outputs=True)
        student_logits = student_outputs.logits

        if self.phase == 'train':
            with torch.no_grad():
                teacher_loss_ce, teacher_outputs = super().compute_loss(self.teacher_model, inputs, return_outputs=True)
                teacher_logits = teacher_outputs.logits

            # Compute the new custom KD loss
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            student_probs = F.softmax(student_logits, dim=-1)

            # Compute teacher2label_dif: absolute difference between teacher's predictions and the correct labels
            correct_labels = torch.nn.functional.one_hot(inputs["labels"], num_classes=teacher_probs.size(-1)).float()
            teacher2label_dif = torch.abs(teacher_probs - correct_labels)

            # Compute student2teacher_dif: absolute difference between student's and teacher's predictions
            student2teacher_dif = torch.abs(student_probs - teacher_probs)

            # Sum the differences across the last dimension (class dimension) before the dot product
            teacher2label_dif_sum = teacher2label_dif.sum(dim=-1)
            student2teacher_dif_sum = student2teacher_dif.sum(dim=-1)

            # Custom KD loss: dot product of teacher2label_dif_sum and student2teacher_dif_sum
            student_loss_kd = torch.dot(teacher2label_dif_sum.view(-1), student2teacher_dif_sum.view(-1))

            # Store the current losses
            self.loss_ce_history.append(student_loss_ce.item())
            self.loss_kd_history.append(student_loss_kd.item())

            if self.one_loss_at_a_time:
                # Alternate between student_loss_ce and student_loss_kd
                if self.alternate:
                    student_combined_loss = student_loss_kd
                else:
                    student_combined_loss = student_loss_ce
                self.alternate = not self.alternate
            else:
                # Calculate differential weights
                if self.args.adaptive_alpha:
                    k = self.args.len_loss_history
                    self.args.alpha, self.args.beta = self._calculate_differential_weights(k)

                # Combined loss with differential weights
                student_combined_loss = self.args.alpha * student_loss_ce + self.args.beta * student_loss_kd

            wandb.log({
                f"{self.phase}_loss": student_combined_loss.item(),
                f"{self.phase}_loss_ce": student_loss_ce.item(),
                f"{self.phase}_loss_kd": student_loss_kd.item(),
                f"{self.phase}_alpha": self.args.alpha,
                f"{self.phase}_beta": self.args.beta,
                f"{self.phase}_teacher_loss": teacher_loss_ce.item(),
                f"{self.phase}_student_perplexity": torch.exp(student_loss_ce).item(),
                f"{self.phase}_teacher_perplexity": torch.exp(teacher_loss_ce).item()
            })

        elif self.phase == 'eval':
            student_combined_loss = student_loss_ce
            wandb.log({
                f"{self.phase}_loss": student_loss_ce.item(),
                f"{self.phase}_loss_ce": student_loss_ce.item(),
                f"{self.phase}_student_perplexity": torch.exp(student_loss_ce).item()
            })

        return (student_combined_loss, student_outputs) if return_outputs else student_combined_loss



    def training_step(self, model, inputs):
        self.phase = 'train'  # Set phase to train at the beginning of each training step
        return super().training_step(model, inputs)

        
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self.phase = 'eval'
        # First, evaluate the student model using the superclass's evaluate method
        student_evaluation_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
    
        # merge or display both the student's and teacher's metrics:
        evaluation_results = {
            **student_evaluation_results,
        }
        self.phase = 'train'
        return evaluation_results


en_student_trainer = KnowledgeDistillationTrainer(
    model=student_model,
    args=student_training_args,
    train_dataset=tokenized_train_datasets[lang2code[selected_language]],
    eval_dataset=tokenized_valid_datasets,
    tokenizer=tokenizer,
    teacher_model=teacher_model,
    one_loss_at_a_time=one_loss_at_a_time,
    teacher_trainer=None#teacher_trainer,
)

print("Start training ... ")
# Use the Accelerator to prepare the trainer
en_student_trainer = accelerator.prepare(en_student_trainer)


try:
    en_student_trainer.train()
except KeyboardInterrupt:
    print("Keyboard interruption detected.")
finally:
    print("Saving the model and training state before exiting...")
    # Specify the directory to save model, tokenizer, and training state
    save_dir = f"./interrupted_{student_name}_{student_model_params_millions}M_training_checkpoint_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    # Ensure the output_dir in training arguments points to the new save directory
    en_student_trainer.args.output_dir = save_dir

    # Saving model and tokenizer using Trainer's convenient method
    en_student_trainer.save_model(save_dir)
    if en_student_trainer.tokenizer is not None:
        en_student_trainer.tokenizer.save_pretrained(save_dir)
    
    # Saving optimizer and scheduler states
    en_student_trainer.save_state()
    
    # Optionally, save other components like the training arguments if needed for resuming
    try:
        training_args_save_path = f"{save_dir}/student_training_args.bin"
        torch.save(en_student_trainer.args, training_args_save_path)
    except:
        print("Failed to save Training Arguments")

    print(f"Model, tokenizer, and training state saved to {save_dir}. Exiting.")

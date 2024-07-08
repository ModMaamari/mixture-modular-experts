from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers.modeling_utils import PreTrainedModel ,PretrainedConfig
from torch.utils.data import IterableDataset, DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model
from transformers.file_utils import is_sagemaker_mp_enabled
from transformers import AutoConfig, AutoModelForCausalLM
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.linear_model import SGDClassifier
from transformers import TrainerCallback
from transformers import AutoTokenizer
from sklearn.pipeline import Pipeline
from torch.nn import CrossEntropyLoss
from accelerate import Accelerator
from datasets import DatasetDict
import torch.nn.functional as F
from datetime import datetime
from tqdm.auto import tqdm
import torch.nn as nn
import pandas as pd
import numpy as np
import itertools
import random
import pickle
import torch
import wandb
import torch
import time
import yaml
import json
import os

class InputIDsToString(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Convert list of ids to NumPy array for efficient processing
        return [' '.join(np.array(input_ids)[np.isin(input_ids, [1, 2], invert=True)].astype(str)) for input_ids in X]
    

class SharedEmbedding(PreTrainedModel):
    def __init__(self, vocab_size, n_embd, init_config):
        super(SharedEmbedding, self).__init__(init_config)
        self.embedding = nn.Embedding(vocab_size, n_embd)

    def forward(self, input_ids):
        # print("Forward of SharedEmbedding")
        return self.embedding(input_ids)


class Router(PreTrainedModel):
    def __init__(self, n_embd, num_experts, init_config):
        super(Router, self).__init__(init_config)
        self.fc = nn.Linear(n_embd, num_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, embeddings):
        # print("Forward of Router")
        logits = self.fc(embeddings)
        return self.softmax(logits)


class GPT2Expert(PreTrainedModel):
    def __init__(self, config, init_config):
        super(GPT2Expert, self).__init__(config)
        self.model = GPT2Model(config)

    def forward(self, inputs_embeds, attention_mask):
        # print("Forward of GPT2Expert")
        return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)


class MoEConfig(PretrainedConfig):
    model_type = "moe_model"

    def __init__(self, vocab_size=32000, n_embd=768, num_experts=4, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.num_experts = num_experts

class MoEModel(PreTrainedModel):
    def __init__(self, config, router, num_experts, vocab_size, n_embd, 
                 used_experts=['en','fr','de','py'], use_common_expert=False, 
                 average_experts=False):
        self.init_config = MoEConfig(
                    vocab_size=32000,
                    n_embd=768,
                    num_experts=4,)
        
        super(MoEModel, self).__init__(self.init_config)

        self.embedding = SharedEmbedding(vocab_size, n_embd, self.init_config)
        self.router = router
        self.experts = nn.ModuleList([GPT2Expert(config, self.init_config) for _ in range(num_experts)])
        self.linear = nn.Linear(n_embd, vocab_size)
        self.config = config
        self.used_experts = used_experts if used_experts is not None else []
        self.use_common_expert = use_common_expert
        self.average_experts = average_experts

        self.code2lang = {0: 'en', 1: 'fr', 2: 'de', 3: 'py'}
        self.lang2code = {v: k for k, v in self.code2lang.items()}
        
        if self.use_common_expert:
            self.common_expert = GPT2Expert(config, self.init_config)
            self.lang2code['cmn'] = 5
        
        self.count = 0

    def forward(self, input_ids, attention_mask, labels=None):
        embeddings = self.embedding(input_ids)
        input_ids_list = input_ids.tolist()
        
        # Get routing probabilities and selected expert
        route_probs = self.router.predict_proba([input_ids_list])[0]
        selected_expert = self.router.predict([input_ids_list])[0]
        expert_outputs_list = []
        # Filter the route_probs and selected_expert based on used_experts
        if self.used_experts:
            valid_expert_indices = [self.lang2code[lang] for lang in self.used_experts]
            valid_route_probs = [route_probs[i] for i in valid_expert_indices]
            selected_expert = valid_expert_indices[valid_route_probs.index(max(valid_route_probs))]

            # Gather the outputs of the used experts
            expert_outputs_list = [self.experts[idx](embeddings, attention_mask).last_hidden_state 
                                for idx in valid_expert_indices] if self.average_experts else \
                                [self.experts[selected_expert](embeddings, attention_mask).last_hidden_state]
        
        # If use_common_expert is True, include the common expert output
        if self.use_common_expert:
            common_expert_output = self.common_expert(embeddings, attention_mask).last_hidden_state
            expert_outputs_list.append(common_expert_output)

        # Calculate the average of the gathered expert outputs
        combined_output = sum(expert_outputs_list) / len(expert_outputs_list)

        # Apply the linear layer to project the hidden states to the vocabulary size
        logits = self.linear(combined_output)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return (loss, logits, route_probs, selected_expert)
        else:
            return (logits, route_probs, selected_expert)


    def save_trained_moe(self, save_directory):

        # Ensure the save directory exists
        os.makedirs(save_directory, exist_ok=True)

        custom_config = {
            'used_experts': self.used_experts,
            'use_common_expert': self.use_common_expert,
            'average_experts': self.average_experts
        }
        custom_config_path = os.path.join(save_directory, 'custom_config.json')
        with open(custom_config_path, 'w') as f:
            json.dump(custom_config, f)

        # Save the Embeddings
        embedding_path = os.path.join(save_directory, f'shared_embeddings.pth')
        torch.save(self.embedding.embedding.state_dict(), embedding_path)
        # Save the experts
        for idx, expert in enumerate(self.experts):
            expert_path = os.path.join(save_directory, f'expert_{idx}')
            expert.model.save_pretrained(expert_path)
        # Save the common expert if used
        if self.use_common_expert:
            common_expert_path = os.path.join(save_directory, 'common_expert')
            self.common_expert.model.save_pretrained(common_expert_path)


    def load_trained_moe(self, load_directory):
        # Load the embeddings
        embedding_path = os.path.join(load_directory, 'shared_embeddings.pth')
        self.embedding.embedding.load_state_dict(torch.load(embedding_path))
        print(f'Embeddings loaded from {embedding_path}')
    
        # Load the experts
        for idx, expert in enumerate(self.experts):
            expert_path = os.path.join(load_directory, f'expert_{idx}')
            expert.model = expert.model.from_pretrained(expert_path)
            print(f'Expert {idx} loaded from {expert_path}')
    
        # Load the common expert if used
        if self.use_common_expert:
            common_expert_path = os.path.join(load_directory, 'common_expert')
            self.common_expert.model = self.common_expert.model.from_pretrained(common_expert_path)
            print(f'Common expert loaded from {common_expert_path}')
        
    
class MultiLangDataset(Dataset):
    def __init__(self, datasets, batch_size):
        self.datasets = datasets
        self.keys = list(datasets.keys())
        self.iterators = {k: iter(itertools.cycle(datasets[k])) for k in self.keys}
        self.batch_size = batch_size * torch.cuda.device_count()
        self.current_key_index = 0
        self.current_count = 0

    def __len__(self):
        return sum(len(d) for d in self.datasets.values())

    def __getitem__(self, index):
        if self.current_count >= self.batch_size:
            self.current_count = 0
            self.current_key_index = (self.current_key_index + 1) % len(self.keys)

        lang_key = self.keys[self.current_key_index]
        self.current_count += 1
        return next(self.iterators[lang_key])
    
# Set CUDA_VISIBLE_DEVICES to only see GPU 0,1
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" #todo 


print("Loading the tokenizer")
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("En_De_Fr_Py_Tokenizer")
tokenizer.pad_token = tokenizer.eos_token

# Initialize the Accelerator
accelerator = Accelerator()
device = accelerator.device
# device = torch.device("cuda:0" if torch.cuda.device_count() > 0 else "cpu")
print(f"Device is {device}")
print("Loading the data")


max_position_embeddings = 1024
code2lang = {0:'en', 1:'fr', 2:'de', 3:'py'}
lang2code = {v:k for k,v in code2lang.items()}

selected_language = 'en'
load_sample=False
drop_language_feature = True
use_common_expert = True
used_experts=['en','fr','de','py']
average_experts=False

load_sample_prefix = '' if not load_sample else '_sample'
with open(f'tokenized_datasets/tokenized_train_datasets_1kContext_sep4lang{load_sample_prefix}.pkl', 'rb') as file:
    shuffle_seed = random.randint(1,999999)
    np.random.seed(shuffle_seed)
    tokenized_train_datasets = pickle.load(file)
    # Shuffle
    tokenized_train_datasets = {key: dataset.shuffle() for key, dataset in tokenized_train_datasets.items()}
    if drop_language_feature:
        tokenized_train_datasets = {key: dataset.remove_columns('language') for key, dataset in tokenized_train_datasets.items()}


# Custom collate function to ensure batches are from one language
def collate_fn(batch):
    input_ids = torch.tensor([item['input_ids'] for item in batch], dtype=torch.long)
    attention_mask = torch.tensor([item['attention_mask'] for item in batch], dtype=torch.long)
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
    # language = torch.tensor([item['language'] for item in batch], dtype=torch.long)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        # 'language': language
    }


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
    n_head=6, #12
    n_layer=3, #6
    n_positions=1024,
    vocab_size=32000,
    resid_pdrop=0.1,
    torch_dtype="bfloat16",
)

# Load the router
with open('language_router_pipeline.pkl', 'rb') as file:
    router = pickle.load(file)

# Initialize the MoE model
moe_model = MoEModel(student_config, router=router, num_experts=4, 
                     vocab_size=vocab_size, n_embd=student_config.__dict__['n_embd'],
                     use_common_expert=use_common_expert, used_experts=used_experts, 
                     average_experts=average_experts)

teacher_model = teacher_model.to(device)
moe_model = moe_model.to(device)

def count_parameters_in_millions(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000

student_model_params_millions = count_parameters_in_millions(moe_model)
student_name = f"MoE GPT2 {student_model_params_millions:.2f}M"
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
output_dir = f"./MoE_KD_4langs_{int(student_model_params_millions)}M_exp_results_{datetime.now().strftime('%Y-%m-%d %H:%M')}"
epochs=3
train_batch_size = 32
eval_batch_size = 64
len_train_data = sum([x.num_rows for x in tokenized_train_datasets.values()]) # tokenized_train_datasets[lang2code[selected_language]].num_rows #todo
num_GPUs = ds_config['num_processes']
num_steps = int(((len_train_data*epochs)/(train_batch_size*GA_steps))/num_GPUs)
eval_steps = num_steps//16
len_loss_history = max(20,num_steps//100)

# Create MultiLangDataset instance
multi_lang_dataset = MultiLangDataset(tokenized_train_datasets, train_batch_size*GA_steps)

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
wandb.config.update({'use_common_expert':use_common_expert})
wandb.config.update({'used_experts':used_experts})
wandb.config.update({'average_experts':average_experts})
wandb.config.update({'student_size (million)':student_model_params_millions})
wandb.config.update({'teacher_size (million)':teacher_model_params_millions})
wandb.config.update({'num_GPUs':num_GPUs})
wandb.config.update({"adaptive_alpha_config": adaptive_alpha_config})
wandb.config.update({"KD_config": KD_config})

# Update wandb config with the deepSpeed config
wandb.config.update({"deep_speed_config": ds_config})
wandb.config.update({"Number of training samples": len_train_data})
wandb.config.update({"Number of training steps": num_steps})
wandb.config.update({"Number of validation samples": len(tokenized_valid_datasets)})
wandb.config.update({"Train dataset shuffle random seed": shuffle_seed})
wandb.save('KD_MoE_training.py')
wandb.save('ds_config.yml')

print("Line 420, before class KnowledgeDistillationTrainer(Trainer):")
class KnowledgeDistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, teacher_trainer=None, **kwargs):
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

    def compute_loss(self, model, inputs, return_outputs=False):
        # Compute loss and outputs for the student model using the superclass method
        student_loss_ce, student_outputs = super().compute_loss(model, inputs, return_outputs=True)
        student_logits = student_outputs[1]

        if self.phase == 'train':
            with torch.no_grad():
                teacher_loss_ce, teacher_outputs = super().compute_loss(self.teacher_model, inputs, return_outputs=True)
                teacher_logits = teacher_outputs.logits
            
            # Compute distillation loss, normal KL or inverse KL based on args
            if not self.args.inverse_KL:
                student_loss_kd = nn.functional.kl_div(
                    input=F.log_softmax(student_logits / self.args.temperature, dim=-1).float(),
                    target=F.softmax(teacher_logits / self.args.temperature, dim=-1).float(),
                    reduction='batchmean'
                ) * (self.args.temperature ** 2)
            else:            
                # Inverse KL
                teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.bfloat16)
                inf_mask = torch.isinf(student_logits)
                logprobs = F.log_softmax(student_logits, dim=-1, dtype=torch.bfloat16)
                prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
                x = torch.sum(prod_probs, dim=-1).view(-1)
                loss_mask = (inputs["input_ids"] != tokenizer.pad_token_id).int().view(-1)
                student_loss_kd = -torch.sum(x * loss_mask, dim=0) / torch.sum(loss_mask, dim=0)
                

            # Store the current losses
            self.loss_ce_history.append(student_loss_ce.item())
            self.loss_kd_history.append(student_loss_kd.item())

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
        
        # Get the input ids to determine the selected expert
        input_ids_list = inputs['input_ids'].tolist()
        selected_expert = model.router.predict([input_ids_list])[0]
        
        # Freeze non-selected experts
        for i, expert in enumerate(model.experts):
            if i == selected_expert:
                expert.train()
                for param in expert.parameters():
                    param.requires_grad = True
            else:
                expert.eval()
                for param in expert.parameters():
                    param.requires_grad = False

        # Ensure the common expert and embeddings are trainable if they are used
        if model.use_common_expert:
            model.common_expert.train()
            for param in model.common_expert.parameters():
                param.requires_grad = True

        model.embedding.train()
        for param in model.embedding.parameters():
            param.requires_grad = True

        return super().training_step(model, inputs)

        
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self.phase = 'eval'
        # First, evaluate the student model using the superclass's evaluate method
        student_evaluation_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
    
        evaluation_results = {
            **student_evaluation_results,
        }
        self.phase = 'train'
        return evaluation_results


moe_trainer = KnowledgeDistillationTrainer(
    model=moe_model,
    args=student_training_args,
    train_dataset=multi_lang_dataset,
    eval_dataset=tokenized_valid_datasets,
    tokenizer=tokenizer,
    teacher_model=teacher_model,
    data_collator=collate_fn,
    teacher_trainer=None#teacher_trainer,
)


print("Start training ... ")
# Use the Accelerator to prepare the trainer
moe_trainer = accelerator.prepare(moe_trainer) #todo

try:
    moe_trainer.train()
    print("Training completed.")
    print("Saving the final model...")
    moe_model.save_trained_moe(os.path.join(output_dir, 'final_model'))
    print(f"Model saved in {os.path.join(output_dir, 'final_model')}")
except KeyboardInterrupt:
    print("Keyboard interruption detected.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    #Evaluate the model
    print("Evaluating the model...")
    # Function to initialize the Trainer and calculate perplexity
    def evaluate_model(model, dataset, device):
        student_training_args.do_train = False
        student_training_args.do_eval = True
        moe_trainer.args = student_training_args
        # moe_trainer = KnowledgeDistillationTrainer(
        #     model=model,
        #     args=student_training_args,
        #     eval_dataset=dataset,
        #     tokenizer=tokenizer,
        #     teacher_model=teacher_model,
        #     data_collator=collate_fn,
        #     teacher_trainer=None#teacher_trainer,
        # )

        eval_result = moe_trainer.evaluate(eval_dataset=dataset)
        print(eval_result)
        # perplexity = torch.exp(torch.tensor(eval_result['eval_loss'])).item()
        # return perplexity
        return eval_result
    
    results_df = pd.DataFrame()

    # List of evaluation settings
    evaluation_settings = [
        {'use_common_expert': False, 'used_experts': ['en', 'fr', 'de', 'py'], 'average_experts': False},
        
        {'use_common_expert': False, 'used_experts': ['en'], 'average_experts': False},
        {'use_common_expert': True, 'used_experts': ['en'], 'average_experts': False},
        
        {'use_common_expert': False, 'used_experts': ['fr'], 'average_experts': False},
        {'use_common_expert': True, 'used_experts': ['fr'], 'average_experts': False},
        
        {'use_common_expert': False, 'used_experts': ['de'], 'average_experts': False},
        {'use_common_expert': True, 'used_experts': ['de'], 'average_experts': False},
        
        {'use_common_expert': False, 'used_experts': ['py'], 'average_experts': False},
        {'use_common_expert': True, 'used_experts': ['py'], 'average_experts': False},

        {'use_common_expert': True, 'used_experts': ['en', 'fr', 'de', 'py'], 'average_experts': False},
        {'use_common_expert': True, 'used_experts': [], 'average_experts': False},

        {'use_common_expert': False, 'used_experts': ['en','py'], 'average_experts': False},
        {'use_common_expert': False, 'used_experts': ['fr','py'], 'average_experts': False},
        {'use_common_expert': False, 'used_experts': ['de','py'], 'average_experts': False},

        {'use_common_expert': False, 'used_experts': ['en','py'], 'average_experts': True},
        {'use_common_expert': False, 'used_experts': ['fr','py'], 'average_experts': True},
        {'use_common_expert': False, 'used_experts': ['de','py'], 'average_experts': True},

        {'use_common_expert': True, 'used_experts': ['en','py'], 'average_experts': False},
        {'use_common_expert': True, 'used_experts': ['fr','py'], 'average_experts': False},
        {'use_common_expert': True, 'used_experts': ['de','py'], 'average_experts': False},

        {'use_common_expert': False, 'used_experts': ['en', 'fr', 'de', 'py'], 'average_experts': True},

    ]

    # Directory to save results
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)

    # Perform evaluations
    timestamp = time.time()

    for setting in evaluation_settings:
        print(f"Evaluating setting: {setting}")
        
        # Update the model settings for the current evaluation
        moe_model.use_common_expert = setting['use_common_expert']
        moe_model.used_experts = setting['used_experts']
        moe_model.average_experts = setting['average_experts']

        # Calculate the number of active parameters
        # num_active_params = calculate_active_params(moe_model, setting['used_experts'], setting['use_common_expert'])
        # num_all_params = calculate_all_params(moe_model)

        result_dict = {**setting}#, "num_active_params": num_active_params, "num_all_params": num_all_params}
        start_time = time.time()
        
        en_eval_result = evaluate_model(moe_model, tokenized_valid_datasets['en'], device)
        result_dict["en_loss"] = en_eval_result['eval_loss']
        result_dict["en_perplexity"] = torch.exp(torch.tensor(en_eval_result['eval_loss'])).item()
    

        fr_eval_result = evaluate_model(moe_model, tokenized_valid_datasets['fr'], device)
        result_dict["fr_loss"] = fr_eval_result['eval_loss']
        result_dict["fr_perplexity"] = torch.exp(torch.tensor(fr_eval_result['eval_loss'])).item()
    
        

        de_eval_result = evaluate_model(moe_model, tokenized_valid_datasets['de'], device)
        result_dict["de_loss"] = de_eval_result['eval_loss']
        result_dict["de_perplexity"] = torch.exp(torch.tensor(de_eval_result['eval_loss'])).item()
        
        

        py_eval_result = evaluate_model(moe_model, tokenized_valid_datasets['py'], device)
        result_dict["py_loss"] = py_eval_result['eval_loss']
        result_dict["py_perplexity"] = torch.exp(torch.tensor(py_eval_result['eval_loss'])).item()        


        # for lang, dataset in tokenized_valid_datasets.items():
        #     # first, get the dataset ready by putting it in a datadict
        #     dataset = DatasetDict({lang2code[lang]: dataset})
        #     perplexity = evaluate_model(moe_model, dataset, device)
        #     result_dict[f"{lang}_perplexity"] = perplexity
        #     print(f"Perplexity for {lang}: {perplexity}")
        
        eval_time = time.time() - start_time
        result_dict["eval_time"] = eval_time
        print(result_dict)
        print('_' * 20)
        results_df = pd.concat([results_df, pd.DataFrame([result_dict])], ignore_index=True)
        results_df.to_csv(f"{results_dir}/moe_model_performance_{timestamp}.csv", index=False)

        # torch.cuda.empty_cache()

    print("All Settings evaluated and results saved.")



    print("Saving the model and training state before exiting...")
    # Specify the directory to save model, tokenizer, and training state
    save_dir = f"./interrupted_{student_name}_{student_model_params_millions}M_training_checkpoint_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    # Ensure the output_dir in training arguments points to the new save directory
    moe_trainer.args.output_dir = save_dir

    # Optionally, save other components like the training arguments if needed for resuming
    try:
        training_args_save_path = f"{save_dir}/student_training_args.bin"
        torch.save(moe_trainer.args, training_args_save_path)
    except:
        print("Failed to save Training Arguments")

    print(f"Model, tokenizer, and training state saved to {save_dir}. Exiting.")
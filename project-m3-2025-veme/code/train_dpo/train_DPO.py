
import argparse
import random
import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import os

from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

import wandb
from tqdm import tqdm


#check if CUDA is avaiable
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.backends.cuda.is_built())


#seed function for all used libraries
def seed_everything(seed=2003):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


#First loss: DPO log sigmoid (as in original implementation)
def calculate_DPO_loss(model_preferred_logprob, model_dispreferred_logprob,
                       ref_preferred_logprob, ref_dispreferred_logprob,
                       beta=0.5):

    preferred_relative_logprob = model_preferred_logprob - ref_preferred_logprob
    dispreferred_relative_logprob = model_dispreferred_logprob - ref_dispreferred_logprob

    reward_accuracies = (preferred_relative_logprob > dispreferred_relative_logprob).float().mean()
    reward_margins = (preferred_relative_logprob - dispreferred_relative_logprob).mean()

    loss = -F.logsigmoid(beta * (preferred_relative_logprob - dispreferred_relative_logprob)).mean()

    return loss, preferred_relative_logprob.mean(), dispreferred_relative_logprob.mean(), reward_accuracies, reward_margins



#normalized version of the log sigmoid loss, not used in final runs but just if needed
def calculate_IPO_loss(model_preferred_logprob, model_dispreferred_logprob,
                       ref_preferred_logprob, ref_dispreferred_logprob,
                       beta=0.5):

    preferred_relative_logprob = model_preferred_logprob - ref_preferred_logprob
    dispreferred_relative_logprob = model_dispreferred_logprob - ref_dispreferred_logprob

    r_diff = beta * (preferred_relative_logprob - dispreferred_relative_logprob)
    r_neg_diff = -r_diff

    logsigmoid_term = -F.logsigmoid(r_diff)
    #adding normalization term to the orignal DPO loss
    normalization_term = torch.log(torch.exp(r_diff) + torch.exp(r_neg_diff))

    loss = (logsigmoid_term + normalization_term).mean()

    reward_accuracies = (preferred_relative_logprob > dispreferred_relative_logprob).float().mean()
    reward_margins = (preferred_relative_logprob - dispreferred_relative_logprob).mean()

    return loss, preferred_relative_logprob.mean(), dispreferred_relative_logprob.mean(), reward_accuracies, reward_margins



def calculate_DPO_hinge_norm_loss(model_preferred_logprob, model_dispreferred_logprob,
                                   ref_preferred_logprob, ref_dispreferred_logprob,
                                   gamma=1.0):
    """
    Implements hinge-norm loss (Eq. 10 from the RSO paper):
    Encourages the normalized log-ratio for the chosen sample to exceed the rejected one by a margin (1/gamma).
    """

    #as before
    chosen_reward = model_preferred_logprob - ref_preferred_logprob
    rejected_reward = model_dispreferred_logprob - ref_dispreferred_logprob

    # Hinge loss: max(0, 1 - gamma(chosen - rejected))
    reward_diff = gamma * (chosen_reward - rejected_reward)
    loss = F.relu(1.0 - reward_diff).mean()

    # Logging metrics
    reward_accuracies = (chosen_reward > rejected_reward).float().mean()
    reward_margins = (chosen_reward - rejected_reward).mean()

    return loss, chosen_reward.mean(), rejected_reward.mean(), reward_accuracies, reward_margins



#utilities functions


#first utility to get log probabilities from model logits
def get_log_prob(logits, labels, prompt_lengths):
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
    
    batch_size, seq_len = labels.shape
    response_mask = torch.arange(seq_len, device=labels.device).unsqueeze(0) >= prompt_lengths.unsqueeze(1)
    response_mask = response_mask.float()
    
    response_log_probs = (token_log_probs * response_mask).sum(dim=-1)
    response_lengths = response_mask.sum(dim=-1).clamp(min=1)
    return response_log_probs / response_lengths


#data collator
def collate_fn(batch, tokenizer, max_length, device):
    prompt_encodings = tokenizer(
        ['Instruct: ' + item['prompt'] + '\n' for item in batch],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    chosen_encodings = tokenizer(
        ['Output: ' + item['chosen'] for item in batch],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    rejected_encodings = tokenizer(
        ['Output: ' + item['rejected'] for item in batch],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

    prompt_preferred_ids = torch.cat([
        prompt_encodings.input_ids,
        chosen_encodings.input_ids
    ], dim=-1).to(device)
    
    prompt_dispreferred_ids = torch.cat([
        prompt_encodings.input_ids,
        rejected_encodings.input_ids
    ], dim=-1).to(device)

    prompt_preferred_mask = torch.cat([
        prompt_encodings.attention_mask,
        chosen_encodings.attention_mask
    ], dim=-1).to(device)
    
    prompt_dispreferred_mask = torch.cat([
        prompt_encodings.attention_mask,
        rejected_encodings.attention_mask
    ], dim=-1).to(device)

    prompt_lengths = prompt_encodings.attention_mask.sum(dim=-1).to(device)

    return {
        'prompt_preferred_ids': prompt_preferred_ids,
        'prompt_dispreferred_ids': prompt_dispreferred_ids,
        'prompt_preferred_mask': prompt_preferred_mask,
        'prompt_dispreferred_mask': prompt_dispreferred_mask,
        'prompt_lengths': prompt_lengths
    }

#KL divergence computation, to compare learned model with reference
def compute_kl_div(model_logits, ref_logits, attention_mask):
    """
    model_logits: [B, T, V]
    ref_logits: [B, T, V]
    attention_mask: [B, T]
    """
    model_log_probs = F.log_softmax(model_logits, dim=-1)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    # KL divergence per token: KL(ref || model)
    kl = F.kl_div(model_log_probs, ref_log_probs.exp(), reduction="none", log_target=False)

    # Sum over vocab, then average over tokens (masked)
    kl_token = kl.sum(-1)
    mask = attention_mask.float()
    kl_mean = (kl_token * mask).sum() / mask.sum()

    return kl_mean



#MAIN TRAINING LOOP

def train(model, ref_model, tokenizer, optimizer, train_dataloader, val_dataloader, epochs=1, beta=0.1,
           loss_name='LogS', gradient_acc_steps = 1, gamma=1.0, save_path='Qwen06B_save_path', 
           steps=500, name="LogS_loss"):
    
    #Validation function to use once the unique epoch (with final dataset) is finished
    def validate():
        model.eval()
        correct, total = 0, 0
        total_margin = 0.0

        with torch.no_grad():
            for batch in val_dataloader:
                model_preferred_logits = model(
                    input_ids=batch['prompt_preferred_ids'],
                    attention_mask=batch['prompt_preferred_mask']
                ).logits

                model_dispreferred_logits = model(
                    input_ids=batch['prompt_dispreferred_ids'],
                    attention_mask=batch['prompt_dispreferred_mask']
                ).logits

                model_preferred_logprob = get_log_prob(
                    model_preferred_logits, batch['prompt_preferred_ids'], batch['prompt_lengths']
                )
                model_dispreferred_logprob = get_log_prob(
                    model_dispreferred_logits, batch['prompt_dispreferred_ids'], batch['prompt_lengths']
                )

                margin = (model_preferred_logprob - model_dispreferred_logprob).detach()
                correct += (margin > 0).float().sum().item()
                total += margin.numel()
                total_margin += margin.sum().item()

        acc = correct / total
        avg_margin = total_margin / total
        print(f"[VAL] reward_accuracy={acc:.4f}  reward_margin={avg_margin:.4f}")

        wandb.log({
            'val_reward_accuracy': acc,
            'val_reward_margin': avg_margin,
        })
        model.train()

    #training loop
    model.train()
    ref_model.eval()
    step = 0
    for epoch in range(epochs):
        for batch in tqdm(train_dataloader):
            step += 1
            optimizer.zero_grad()

            #get DPO model logits and logprobs for preferred and dispreferred answers
            model_preferred_logits = model(
                input_ids=batch['prompt_preferred_ids'],
                attention_mask=batch['prompt_preferred_mask']
            ).logits

            model_preferred_logprob = get_log_prob(
                model_preferred_logits,
                batch['prompt_preferred_ids'],
                batch['prompt_lengths']
            )

            model_dispreferred_logits = model(
                input_ids=batch['prompt_dispreferred_ids'],
                attention_mask=batch['prompt_dispreferred_mask']
            ).logits
            
            model_dispreferred_logprob = get_log_prob(
                model_dispreferred_logits,
                batch['prompt_dispreferred_ids'],
                batch['prompt_lengths']
            )

            #now get them for reference model (frozen)
            with torch.no_grad():

                ref_preferred_logits = ref_model(
                    input_ids=batch['prompt_preferred_ids'],
                    attention_mask=batch['prompt_preferred_mask']
                ).logits
                
                ref_preferred_logprob = get_log_prob(
                    ref_preferred_logits,
                    batch['prompt_preferred_ids'],
                    batch['prompt_lengths']
                )

                ref_dispreferred_logits = ref_model(
                    input_ids=batch['prompt_dispreferred_ids'],
                    attention_mask=batch['prompt_dispreferred_mask']
                ).logits
                
                ref_dispreferred_logprob = get_log_prob(
                    ref_dispreferred_logits,
                    batch['prompt_dispreferred_ids'],
                    batch['prompt_lengths']
                )

                #now get KL divergencies for preferred and dispreferred answers

                kl_preferred = compute_kl_div(model_preferred_logits, ref_preferred_logits, batch["prompt_preferred_mask"])
                kl_dispreferred = compute_kl_div(model_dispreferred_logits, ref_dispreferred_logits, batch["prompt_dispreferred_mask"])

            #loss computation, choice between logSigmoid, IPO (implemented as normalized logs), Hinge from RSO

            if loss_name == 'LogS':
                loss, preferred_relative_logprob, dispreferred_relative_logprob, reward_accuracies, reward_margins = calculate_DPO_loss(
                    model_preferred_logprob,
                    model_dispreferred_logprob,
                    ref_preferred_logprob,
                    ref_dispreferred_logprob,
                    beta=beta
                )
            elif loss_name == 'IPO':
                loss, preferred_relative_logprob, dispreferred_relative_logprob, reward_accuracies, reward_margins = calculate_IPO_loss(
                    model_preferred_logprob,
                    model_dispreferred_logprob,
                    ref_preferred_logprob,
                    ref_dispreferred_logprob,
                    beta=beta
                )
            elif loss_name == 'hinge':
                loss, preferred_relative_logprob, dispreferred_relative_logprob, reward_accuracies, reward_margins = calculate_DPO_hinge_norm_loss(
                    model_preferred_logprob,
                    model_dispreferred_logprob,
                    ref_preferred_logprob,
                    ref_dispreferred_logprob,
                    gamma = gamma
                )
            
            #As gradient accumulation is used, loss is divided by accumulation steps
            loss = loss / gradient_acc_steps
            loss.backward()

            #update weights every 'gradient_acc_steps'
            if step % gradient_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            #save model checkpoint (only one that overwrites previous ones) every 'steps' steps
            if step % steps == 0 and step > 0:
                checkpoint_dir = os.path.join(save_path, name + str(steps))
                model.save_pretrained(checkpoint_dir, safe_serialization=True)
                tokenizer.save_pretrained(checkpoint_dir)
                print(f"Overwrote checkpoint at step {step} to {checkpoint_dir} / {name}")

            #log metrics in wandb
            wandb.log({
                'loss': loss.item() * gradient_acc_steps, #scale back the loss
                'preferred_relative_logprob': preferred_relative_logprob.item(),
                'dispreferred_relative_logprob': dispreferred_relative_logprob.item(),
                'reward_accuracy': reward_accuracies.item(),
                'reward_margin': reward_margins.item(),
                "kl_preferred": kl_preferred.item(),
                "kl_dispreferred": kl_dispreferred.item(),
            })

    #if validation is enabled, validate once training is finished
    if val_dataloader:
        validate()

def main():

    #disabling triton and flash attention to avoid problems
    import os
    os.environ["DISABLE_TRITON"] = "1"
    os.environ["FLASH_ATTENTION_DISABLE"] = "1"
    
    #getting model configuration
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=2003)
    parser.add_argument("--model_name", type=str, default="MoroM02/MoroM02") #put sft model as default
    parser.add_argument("--dataset_name", type=str, default="MoroM02/MNLP_M3_dpo_dataset") #put final dataset as default
    parser.add_argument("--wandb_project", type=str, default="Qwen-dpo")
    parser.add_argument("--loss", type=str, default="LogS")
    parser.add_argument("--grad_steps", type=int, default=1)
    parser.add_argument("--gamma_hinge", type=float, default=1.0)
    parser.add_argument("--save_path", type=str, default="checkpoints/", help="Path to save model checkpoints")
    parser.add_argument("--save_every", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--run_name", type=str, default="qwen_06B")


    args = parser.parse_args()

    seed_everything(args.seed)

    wandb.login()
    wandb.init(project=args.wandb_project, config=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(">>> LOADING TOKENIZER")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizer size after pad token:", len(tokenizer))

    print(">>> LOADING MODEL")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name,torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

    model.resize_token_embeddings(len(tokenizer))
    ref_model.resize_token_embeddings(len(tokenizer))

    #adapting tokenizer size if needed
    with torch.no_grad():
        num_new_tokens = len(tokenizer) - model.get_input_embeddings().weight.shape[0]
        if num_new_tokens > 0:
            print(f"Reinitializing {num_new_tokens} new tokens...")
            model.resize_token_embeddings(len(tokenizer))  # Ensure again just in case
            ref_model.resize_token_embeddings(len(tokenizer))
            model.get_input_embeddings().weight[-num_new_tokens:] = torch.nn.init.normal_(
                torch.empty_like(model.get_input_embeddings().weight[-num_new_tokens:]), 0.0, 0.02
            )
            ref_model.get_input_embeddings().weight[-num_new_tokens:] = model.get_input_embeddings().weight[-num_new_tokens:]

    # Sanity check
    print("Model embedding size:", model.get_input_embeddings().weight.shape[0])

    #freeze reference model
    ref_model.requires_grad_(False)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    print(">>> LOADING DATA")
    collate = partial(collate_fn, tokenizer=tokenizer, max_length=args.max_length, device=device)
    dataset = load_dataset(args.dataset_name)
    train_data = dataset['train']
    val_data = dataset['val']
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    print("STARTING TRAINING")
    train(model, ref_model, tokenizer, optimizer, train_dataloader, val_dataloader, epochs=args.epochs, beta=args.beta,
           loss_name=args.loss, gradient_acc_steps=args.grad_steps, gamma=args.gamma_hinge, save_path=args.save_path, 
           steps=args.save_every, name=args.run_name)

    #push models on HF and save locally - comment if not required
    model.save_pretrained(args.run_name, safe_serialization=False)
    model.push_to_hub("MoroM02/MNLP_M2_dpo_model_logs_1ep_fd")
    tokenizer.push_to_hub("MoroM02/MNLP_M2_dpo_model_logs_1ep_fd")

if __name__ == "__main__":
    main()

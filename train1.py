import os
import time
import argparse
import math
import torch
from transformers import AutoTokenizer
from model import DeepSeek, DeepSeekConfig


# -----------------------------------------------------------------------------
# Simple DataLoader
# -----------------------------------------------------------------------------
class DataLoaderLite:
    def __init__(self, B, T, vocab_size, input_file="input.txt"):
        self.B = B
        self.T = T
        self.vocab_size = vocab_size

        with open(input_file, "r") as f:
            text = f.read()

        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        
        # Check tokenizer vocab size
        tokenizer_vocab_size = len(tokenizer)
        print(f"Tokenizer vocabulary size: {tokenizer_vocab_size}")
        print(f"Model vocabulary size: {vocab_size}")
        
        if tokenizer_vocab_size != vocab_size:
            print(f"WARNING: Vocabulary size mismatch!")
            print(f"  Tokenizer has {tokenizer_vocab_size} tokens")
            print(f"  Model expects {vocab_size} tokens")
        
        tokens = tokenizer.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        
        # Filter out tokens that are >= vocab_size
        valid_mask = self.tokens < vocab_size
        invalid_count = (~valid_mask).sum().item()
        
        if invalid_count > 0:
            print(f"Warning: Found {invalid_count} tokens >= vocab_size ({vocab_size})")
            print(f"Clamping invalid tokens to vocab_size - 1")
            # Clamp invalid tokens to maximum valid token ID
            self.tokens = torch.clamp(self.tokens, max=vocab_size - 1)
        
        print(f"Loaded {len(self.tokens)} valid tokens")
        print(f"Token range: [{self.tokens.min().item()}, {self.tokens.max().item()}]")
        print(f"~{len(self.tokens) // (B*T)} batches per full pass")

        self.position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.position: self.position + (B * T + 1)]

        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.position += B * T
        if self.position + (B * T + 1) > len(self.tokens):
            self.position = 0

        return x, y


# -----------------------------------------------------------------------------
# Learning Rate Schedule
# -----------------------------------------------------------------------------
def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Cosine decay with warmup"""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    
    if step > max_steps:
        return min_lr
    
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# -----------------------------------------------------------------------------
# Checkpoints
# -----------------------------------------------------------------------------
def save_checkpoint(model, optimizer, step, loss, filepath, lr_config=None):
    ckpt = {
        "step": step,
        "loss": loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": model.config,
        "lr_config": lr_config,
    }
    torch.save(ckpt, filepath)
    print(f"\nSaved checkpoint: {filepath}")


def load_checkpoint(filepath, device, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        print("Optimizer state entries:", len(optimizer.state_dict()['state']))

    step = checkpoint['step']
    loss = checkpoint['loss']
    lr_config = checkpoint.get('lr_config', None)

    print(f"\nCheckpoint loaded from {filepath}")
    print(f"Resuming from step {step} | loss {loss:.4f}")
    if lr_config:
        print(f"Original LR schedule: max_steps={lr_config['original_max_steps']}")

    return step, loss, lr_config


# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
def train(total_steps=10000, ckpt_path=None, save_path="deepseek_checkpoint.pt", 
          use_lr_schedule=True, log_interval=100, vocab_size=None):
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    torch.set_float32_matmul_precision("high")
    
    # Auto-detect vocab size from tokenizer if not specified
    if vocab_size is None:
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        vocab_size = len(tokenizer)
        print(f"Auto-detected vocabulary size: {vocab_size}")

    # DeepSeek config - exactly as specified
    config = DeepSeekConfig(
        vocab_size=vocab_size,  # Use detected or specified vocab size
        n_layer=30,
        n_head=9,
        n_embd=576,
        head_dim=64,
        intermediate_size=1536,
        max_seq_length=512,
        # MLA parameters
        compression_ratio=8,  # kv_lora_rank = 576 // 8 = 72
        # MoE parameters
        n_routed_experts=8,
        n_shared_experts=1,
        top_k_experts=2,
    )

    model = DeepSeek(config).to(device)
    
    # Print model info
    n_params = model.count_parameters()
    kv_lora_rank = config.n_embd // config.compression_ratio
    
    print(f"\n{'='*70}")
    print(f"DeepSeek Model Configuration")
    print(f"{'='*70}")
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"Vocabulary size: {config.vocab_size:,}")
    print(f"Layers: {config.n_layer}")
    print(f"Heads: {config.n_head}")
    print(f"Embedding dimension: {config.n_embd}")
    print(f"Head dimension: {config.head_dim}")
    print(f"Intermediate size: {config.intermediate_size}")
    print(f"Max sequence length: {config.max_seq_length}")
    print(f"\nMLA Configuration:")
    print(f"  Compression ratio: {config.compression_ratio}")
    print(f"  KV latent rank: {kv_lora_rank}")
    print(f"  RoPE dimension: {config.head_dim // 2}")
    print(f"\nMoE Configuration:")
    print(f"  Total experts: {config.n_routed_experts + config.n_shared_experts} ({config.n_routed_experts} routed + {config.n_shared_experts} shared)")
    print(f"  Top-K routing: {config.top_k_experts}")
    print(f"  Load balancing: DeepSeek V3 Loss-less (bias-based)")
    print(f"{'='*70}\n")
    
    # LR schedule params
    max_lr = 3e-4
    min_lr = max_lr * 0.1
    warmup_steps = 100
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # Data loader - pass vocab_size to validate tokens
    loader = DataLoaderLite(B=4, T=256, vocab_size=config.vocab_size)

    # Load checkpoint if provided
    start_step = 0
    original_max_steps = total_steps
    loaded_lr_config = None
    
    if ckpt_path and os.path.exists(ckpt_path):
        start_step, _, loaded_lr_config = load_checkpoint(ckpt_path, device, model, optimizer)
        
        if loaded_lr_config and 'original_max_steps' in loaded_lr_config:
            original_max_steps = loaded_lr_config['original_max_steps']
            print(f"Continuing with original LR schedule (original max_steps: {original_max_steps})")

    lr_config = {
        'original_max_steps': original_max_steps,
        'max_lr': max_lr,
        'min_lr': min_lr,
        'warmup_steps': warmup_steps,
    }

    print(f"Training Configuration:")
    print(f"  Total steps: {start_step} → {total_steps}")
    print(f"  LR schedule: {'enabled' if use_lr_schedule else 'disabled'} (max={max_lr}, min={min_lr})")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Batch size: {loader.B}")
    print(f"  Sequence length: {loader.T}")
    print(f"  Log interval: {log_interval}\n")

    # Main training loop
    for step in range(start_step, total_steps):
        t0 = time.time()

        # Update learning rate
        if use_lr_schedule:
            lr = get_lr(step, warmup_steps, original_max_steps, max_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = optimizer.param_groups[0]['lr']

        x, y = loader.next_batch()
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if device == "cuda":
            torch.cuda.synchronize()

        t1 = time.time()
        tok_per_sec = (loader.B * loader.T) / (t1 - t0)
        dt = t1 - t0

        if step % log_interval == 0:
            # Estimate MFU (model flops utilization)
            mfu = model.estimate_mfu(loader.B, dt)
            print(f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.6f} | tok/s {tok_per_sec:8.1f} | MFU {mfu*100:.2f}%")
        
        # Save checkpoint and show expert statistics every 1000 steps
        if step % 1000 == 0 and step > 0:
            # Display expert usage statistics
            print(f"\n{'='*70}")
            print(f"Expert Usage Statistics at Step {step}")
            print(f"{'='*70}")
            expert_stats = model.get_expert_statistics()
            
            # Show stats for first and last layer as examples
            for layer_name in ['layer_0', f'layer_{config.n_layer-1}']:
                if layer_name in expert_stats:
                    stats = expert_stats[layer_name]
                    print(f"\n{layer_name.replace('_', ' ').title()}:")
                    print(f"  Expert usage: {stats['usage_percentage']}")
                    print(f"  Expert bias: {stats['bias']}")
            print(f"{'='*70}\n")
            
            ckpt_name = f"deepseek_checkpoint_step_{step}.pt"
            save_checkpoint(model, optimizer, step, loss.item(), ckpt_name, lr_config)

    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"Final loss: {loss.item():.4f}")
    print(f"{'='*70}\n")
    save_checkpoint(model, optimizer, total_steps, loss.item(), save_path, lr_config)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepSeek with MLA and MoE")

    parser.add_argument("--steps", type=int, default=10000,
                        help="Total steps to train to (default: 10000)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint to resume from")
    parser.add_argument("--save", type=str, default="deepseek_checkpoint.pt",
                        help="Where to save final checkpoint")
    parser.add_argument("--no-lr-schedule", action="store_true",
                        help="Disable learning rate schedule (use fixed LR)")
    parser.add_argument("--log-interval", type=int, default=100,
                        help="How often to print training logs")
    parser.add_argument("--vocab-size", type=int, default=None,
                        help="Vocabulary size (auto-detected from tokenizer if not specified)")

    args = parser.parse_args()

    train(
        total_steps=args.steps,
        ckpt_path=args.ckpt,
        save_path=args.save,
        use_lr_schedule=not args.no_lr_schedule,
        log_interval=args.log_interval,
        vocab_size=args.vocab_size
    )

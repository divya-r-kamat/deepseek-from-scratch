import os
import time
import argparse
import math
import torch
from transformers import AutoTokenizer
from deepseek_model import DeepSeek, DeepSeekConfig


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
          use_lr_schedule=True, log_interval=100, vocab_size=None, 
          micro_batch_size=1, gradient_accumulation_steps=4,
          use_original_config=False):
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Enable memory optimizations
    torch.set_float32_matmul_precision("high")
    
    # Enable gradient checkpointing and other memory optimizations
    if device == "cuda":
        torch.cuda.empty_cache()
        # Set memory allocation config for better fragmentation handling
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Auto-detect vocab size from tokenizer if not specified
    if vocab_size is None:
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        vocab_size = len(tokenizer)
        print(f"Auto-detected vocabulary size: {vocab_size}")

    # Choose config based on flag
    if use_original_config:
        print("\n⚠️  WARNING: Using ORIGINAL config - requires significant memory optimizations!")
        print("   This will use: micro_batch=1, seq_len=64, gradient checkpointing, 8-bit optimizer\n")
        
        config = DeepSeekConfig(
            vocab_size=vocab_size,
            n_layer=30,              # ORIGINAL
            n_head=9,                # ORIGINAL
            n_embd=576,              # ORIGINAL
            head_dim=64,
            intermediate_size=1536,  # ORIGINAL
            max_seq_length=512,
            compression_ratio=8,
            n_routed_experts=8,      # ORIGINAL
            n_shared_experts=1,
            top_k_experts=2,
        )
        # Force extreme memory settings for original config
        micro_batch_size = 1
        sequence_length = 64  # Very short sequences
        gradient_accumulation_steps = max(gradient_accumulation_steps, 16)  # More accumulation
    else:
        print("\n✓ Using MEMORY-OPTIMIZED config for 15GB GPU")
        config = DeepSeekConfig(
            vocab_size=vocab_size,
            n_layer=12,              # Reduced
            n_head=6,                # Reduced
            n_embd=384,              # Reduced
            head_dim=64,
            intermediate_size=1024,  # Reduced
            max_seq_length=512,
            compression_ratio=8,
            n_routed_experts=4,      # Reduced
            n_shared_experts=1,
            top_k_experts=2,
        )
        sequence_length = 128

    model = DeepSeek(config).to(device)

    # Print model info
    n_params = model.count_parameters()
    kv_lora_rank = config.n_embd // config.compression_ratio
    
    print(f"\n{'='*70}")
    print(f"DeepSeek Model Configuration {'(ORIGINAL)' if use_original_config else '(OPTIMIZED)'}")
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
    print(f"\nMemory Optimization:")
    print(f"  Micro batch size: {micro_batch_size}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {micro_batch_size * gradient_accumulation_steps}")
    print(f"{'='*70}\n")
    
    # Calculate actual memory usage
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        model_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"Model memory: {model_memory:.2f} GB")
    
    # LR schedule params
    max_lr = 3e-4
    min_lr = max_lr * 0.1
    warmup_steps = 100
    
    # Try to use 8-bit Adam optimizer to save memory (~50% optimizer memory)
    optimizer_name = "AdamW"
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=max_lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
        optimizer_name = "AdamW8bit"
        print(f"✓ Using 8-bit {optimizer_name} optimizer (saves ~50% memory)")
    except ImportError:
        if use_original_config:
            print("\n⚠️  WARNING: bitsandbytes not available!")
            print("   Original config STRONGLY recommends 8-bit optimizer")
            print("   Install with: !pip install bitsandbytes")
            print("   Continuing with standard AdamW (may run out of memory)...\n")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=max_lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )

    # Data loader
    loader = DataLoaderLite(B=micro_batch_size, T=sequence_length, vocab_size=config.vocab_size)

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
    print(f"  Micro batch size: {loader.B}")
    print(f"  Sequence length: {loader.T}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps} steps")
    print(f"  Optimizer: {optimizer_name}")
    print(f"  Log interval: {log_interval}\n")

    # Main training loop with gradient accumulation
    for step in range(start_step, total_steps):
        t0 = time.time()

        # Update learning rate
        if use_lr_schedule:
            lr = get_lr(step, warmup_steps, original_max_steps, max_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = optimizer.param_groups[0]['lr']

        # Gradient accumulation loop
        optimizer.zero_grad()
        loss_accum = 0.0
        
        for micro_step in range(gradient_accumulation_steps):
            x, y = loader.next_batch()
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
                # Scale loss by accumulation steps
                loss = loss / gradient_accumulation_steps
            
            loss_accum += loss.item()
            loss.backward()
            
            # Clear memory after each micro-batch for original config
            if use_original_config:
                del logits
                if device == "cuda":
                    torch.cuda.empty_cache()

        # Clip gradients and step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if device == "cuda":
            torch.cuda.synchronize()

        t1 = time.time()
        # Calculate tokens per second for entire accumulated batch
        tok_per_sec = (loader.B * loader.T * gradient_accumulation_steps) / (t1 - t0)
        dt = t1 - t0

        if step % log_interval == 0:
            # Estimate MFU and show memory usage
            mfu = model.estimate_mfu(loader.B * gradient_accumulation_steps, dt)
            
            if device == "cuda":
                mem_used = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                mem_peak = torch.cuda.max_memory_allocated() / 1024**3
                print(f"step {step:5d} | loss {loss_accum:.4f} | lr {lr:.6f} | tok/s {tok_per_sec:7.1f} | "
                      f"MFU {mfu*100:.1f}% | mem {mem_used:.1f}G (peak {mem_peak:.1f}G)")
            else:
                print(f"step {step:5d} | loss {loss_accum:.4f} | lr {lr:.6f} | tok/s {tok_per_sec:8.1f} | MFU {mfu*100:.2f}%")
        
        # Save checkpoint and show expert statistics every 1000 steps
        if step % 1000 == 0 and step > 0:
            # Clear cache before checkpoint
            if device == "cuda":
                torch.cuda.empty_cache()
            
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
            save_checkpoint(model, optimizer, step, loss_accum, ckpt_name, lr_config)

    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"Final loss: {loss_accum:.4f}")
    print(f"{'='*70}\n")
    save_checkpoint(model, optimizer, total_steps, loss_accum, save_path, lr_config)


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
    parser.add_argument("--micro-batch-size", type=int, default=1,
                        help="Micro batch size per forward pass")
    parser.add_argument("--grad-accum-steps", type=int, default=4,
                        help="Gradient accumulation steps (effective_batch = micro_batch * grad_accum)")
    parser.add_argument("--original-config", action="store_true",
                        help="Use ORIGINAL config (30L, 576d, 8 experts) - requires 8-bit optimizer!")

    args = parser.parse_args()

    train(
        total_steps=args.steps,
        ckpt_path=args.ckpt,
        save_path=args.save,
        use_lr_schedule=not args.no_lr_schedule,
        log_interval=args.log_interval,
        vocab_size=args.vocab_size,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        use_original_config=args.original_config
    )


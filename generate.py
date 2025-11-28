import torch
import argparse
import os
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from model import DeepSeek, DeepSeekConfig


def download_model_from_hf(repo_id, filename, cache_dir="./model_cache"):
    """Download model from Hugging Face Hub"""
    print(f"Downloading model from Hugging Face...")
    print(f"Repository: {repo_id}")
    print(f"File: {filename}")
    
    try:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="space",
            cache_dir=cache_dir
        )
        print(f"✓ Model downloaded successfully!")
        print(f"Location: {model_path}\n")
        return model_path
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        raise


def dequantize_int8_checkpoint(ckpt_path, device):
    """Dequantize INT8 checkpoint back to float32"""
    print(f"Loading quantized checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    quant_state = checkpoint['quant_state']
    meta = checkpoint['meta']
    
    print("Dequantizing model weights...")
    dequant_state = {}
    
    for k, v in quant_state.items():
        if k not in meta:
            # Fallback: keep as-is
            dequant_state[k] = v
            continue
        
        m = meta[k]
        
        if m['type'] == 'int8':
            # Dequantize: multiply by scale
            scale = m['scale']
            dequant = v.float() * scale
            dequant_state[k] = dequant
        elif m['type'] == 'non_tensor':
            dequant_state[k] = v
        else:
            # Raw tensors (already in original dtype)
            dequant_state[k] = v
    
    print("✓ Dequantization complete!\n")
    
    return {
        'model_state': dequant_state,
        'config': checkpoint.get('config'),
        'loss': checkpoint.get('loss'),
        'step': checkpoint.get('step'),
    }


def load_model_from_checkpoint(checkpoint, device):
    """Load model from dequantized checkpoint"""
    config = checkpoint['config']
    
    print(f"Initializing model...")
    model = DeepSeek(config).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    print(f"✓ Model loaded successfully!")
    print(f"Training step: {checkpoint['step']}")
    print(f"Training loss: {checkpoint['loss']:.4f}")
    print(f"Vocabulary size: {config.vocab_size}\n")
    
    return model, config


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=0.9):
    """
    Generate text from the model
    
    Args:
        model: The DeepSeek model
        idx: Starting tokens (B, T) tensor
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: If set, only sample from top k tokens
        top_p: Nucleus sampling threshold
    """
    device = idx.device
    
    for _ in range(max_new_tokens):
        # Crop context if it exceeds max sequence length
        idx_cond = idx if idx.size(1) <= model.config.max_seq_length else idx[:, -model.config.max_seq_length:]
        
        # Forward pass
        with torch.autocast(device_type=device.type if device.type != 'mps' else 'cpu', dtype=torch.bfloat16):
            logits, _ = model(idx_cond)
        
        # Get logits for last position
        logits = logits[:, -1, :] / temperature
        
        # Optional top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        # Optional nucleus (top-p) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least one token
            sorted_indices_to_remove[..., 0] = False
            
            # Scatter to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')
        
        # Sample from distribution
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        idx = torch.cat([idx, idx_next], dim=1)
    
    return idx


def generate_samples(
    repo_id="dkamat/deepseek-from-stratch",
    filename="deepseek_model.pt",
    num_samples=5,
    prompt="",
    max_new_tokens=200,
    temperature=0.8,
    top_k=200,
    top_p=0.9,
    seed=None,
    use_local_checkpoint=None
):
    """Generate multiple text samples from the trained model"""
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{'='*80}")
    print(f"DeepSeek Text Generation from Hugging Face")
    print(f"{'='*80}")
    print(f"Using device: {device}\n")
    
    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        print(f"Random seed set to: {seed}\n")
    
    # Download or load model
    if use_local_checkpoint:
        print(f"Using local checkpoint: {use_local_checkpoint}\n")
        ckpt_path = use_local_checkpoint
    else:
        ckpt_path = download_model_from_hf(repo_id, filename)
    
    # Dequantize and load model
    checkpoint = dequantize_int8_checkpoint(ckpt_path, device)
    model, config = load_model_from_checkpoint(checkpoint, device)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    print("✓ Tokenizer loaded\n")
    
    # Encode prompt
    if prompt:
        prompt_tokens = tokenizer.encode(prompt)
        # Clamp tokens to valid range
        prompt_tokens = [min(t, config.vocab_size - 1) for t in prompt_tokens]
        context = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        print(f"Prompt: '{prompt}'")
    else:
        # Start with a random token if no prompt
        context = torch.randint(0, config.vocab_size, (1, 1), dtype=torch.long, device=device)
        print("No prompt provided, starting from random token")
    
    print(f"\nGenerating {num_samples} samples...")
    print(f"Parameters: max_tokens={max_new_tokens}, temp={temperature}, top_k={top_k}, top_p={top_p}")
    print("="*80)
    
    # Generate samples
    samples = []
    for i in range(num_samples):
        print(f"\n{'='*80}")
        print(f"SAMPLE {i+1}/{num_samples}")
        print(f"{'='*80}")
        
        # Generate
        output = generate(
            model, 
            context.clone(),  # Clone to avoid modifying original
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        # Decode
        generated_tokens = output[0].tolist()
        generated_text = tokenizer.decode(generated_tokens)
        
        samples.append(generated_text)
        print(generated_text)
        print(f"\n{'='*80}")
    
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from quantized DeepSeek model on Hugging Face")
    
    parser.add_argument("--repo-id", type=str, default="dkamat/deepseek-from-stratch",
                        help="Hugging Face repository ID (default: dkamat/deepseek-from-stratch)")
    parser.add_argument("--filename", type=str, default="deepseek_model.pt",
                        help="Model filename in repository (default: deepseek_model.pt)")
    parser.add_argument("--local-checkpoint", type=str, default=None,
                        help="Use local checkpoint instead of downloading (optional)")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of samples to generate (default: 5)")
    parser.add_argument("--prompt", type=str, default="",
                        help="Starting prompt for generation (default: empty)")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Maximum tokens to generate (default: 200)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8)")
    parser.add_argument("--top-k", type=int, default=200,
                        help="Top-k sampling parameter (default: 200)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Nucleus sampling parameter (default: 0.9)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: None)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save samples to file (default: print only)")
    
    args = parser.parse_args()
    
    # Generate samples
    samples = generate_samples(
        repo_id=args.repo_id,
        filename=args.filename,
        num_samples=args.num_samples,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        use_local_checkpoint=args.local_checkpoint
    )
    
    # Optionally save to file
    if args.output:
        with open(args.output, 'w') as f:
            for i, sample in enumerate(samples, 1):
                f.write(f"{'='*80}\n")
                f.write(f"SAMPLE {i}\n")
                f.write(f"{'='*80}\n")
                f.write(sample)
                f.write(f"\n\n")
        print(f"\n✓ Samples saved to {args.output}")

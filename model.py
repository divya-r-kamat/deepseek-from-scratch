import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.weight * x_norm


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim, max_seq_len=8192, theta=100000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        freqs_cis = self._precompute_freqs_cis()
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
    
    def _precompute_freqs_cis(self):
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis
    
    def forward(self, x, start_pos=0):
        batch, n_heads, seq_len, head_dim = x.shape
        freqs_cis = self.freqs_cis[start_pos:start_pos + seq_len]
        
        x_reshaped = x.float().reshape(batch, n_heads, seq_len, -1, 2)
        x_complex = torch.view_as_complex(x_reshaped)
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)
        x_rotated = x_complex * freqs_cis
        x_out = torch.view_as_real(x_rotated)
        x_out = x_out.reshape(batch, n_heads, seq_len, head_dim)
        
        return x_out.type_as(x)


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) - DeepSeek's innovation
    
    Uses compression_ratio to compress KV cache:
    - Compressed dimension = n_embd // compression_ratio
    - Applies RoPE to half of head_dim
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd
        
        # MLA-specific dimensions
        self.kv_lora_rank = config.n_embd // config.compression_ratio  # Compressed KV dimension
        self.qk_rope_dim = config.head_dim // 2    # Half of head_dim for RoPE
        self.qk_nope_dim = self.head_dim - self.qk_rope_dim  # Non-RoPE dimension
        
        # Query projection: full dimension
        self.q_proj = nn.Linear(config.n_embd, config.n_head * config.head_dim, bias=False)
        
        # Compressed KV projection: project to low-rank latent space
        self.kv_a_proj_with_mqa = nn.Linear(
            config.n_embd,
            self.kv_lora_rank + self.qk_rope_dim,
            bias=False
        )
        
        # Per-head decompression of latent KV
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            config.n_head * self.qk_nope_dim,
            bias=False
        )
        
        # Output projection
        self.o_proj = nn.Linear(config.n_head * config.head_dim, config.n_embd, bias=False)
        
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        
        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.max_seq_length, config.max_seq_length))
                .view(1, 1, config.max_seq_length, config.max_seq_length)
        )
        
        # RoPE only for the rope dimension
        self.rope = RotaryPositionalEmbedding(
            self.qk_rope_dim,
            config.max_seq_length,
            config.rope_theta
        )
        
    def forward(self, x):
        B, T, C = x.size()
        
        # Project to queries
        q = self.q_proj(x)  # (B, T, n_head * head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Compress KV to latent space
        compressed_kv = self.kv_a_proj_with_mqa(x)  # (B, T, kv_lora_rank + qk_rope_dim)
        
        # Split into latent KV and rope component
        kv_latent = compressed_kv[:, :, :self.kv_lora_rank]  # (B, T, kv_lora_rank)
        k_rope = compressed_kv[:, :, self.kv_lora_rank:]     # (B, T, qk_rope_dim)
        
        # Decompress latent KV to per-head representations
        kv = self.kv_b_proj(kv_latent)  # (B, T, n_head * qk_nope_dim)
        kv = kv.view(B, T, self.n_head, self.qk_nope_dim).transpose(1, 2)
        
        # Prepare k_rope for concatenation
        k_rope = k_rope.unsqueeze(1).expand(B, self.n_head, T, self.qk_rope_dim)
        
        # Split query into rope and non-rope parts
        q_nope = q[:, :, :, :self.qk_nope_dim]  # (B, n_head, T, qk_nope_dim)
        q_rope = q[:, :, :, self.qk_nope_dim:]  # (B, n_head, T, qk_rope_dim)
        
        # Apply RoPE only to rope components
        q_rope = self.rope(q_rope)
        k_rope = self.rope(k_rope)
        
        # Concatenate rope and non-rope parts
        q = torch.cat([q_nope, q_rope], dim=-1)  # (B, n_head, T, head_dim)
        k = torch.cat([kv, k_rope], dim=-1)      # (B, n_head, T, head_dim)
        v = torch.cat([kv, torch.zeros_like(k_rope)], dim=-1)  # Use kv for values
        
        # Standard attention computation
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        y = self.o_proj(y)
        
        return y


class Expert(nn.Module):
    """Single expert in MoE - SwiGLU FFN"""
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts (MoE) with Shared Experts and DeepSeek V3 Loss-less Load Balancing
    
    Architecture:
    - N routed experts (selected via top-k routing)
    - M shared experts (always active)
    - Loss-less load balancing via bias term
    
    DeepSeek V3 Innovation:
    Instead of auxiliary loss, uses a learnable bias term that encourages load balancing
    without adding to the training loss. The bias is computed from expert usage statistics
    and applied during routing to naturally balance expert selection.
    
    Output = SharedExperts(x) + sum(RouteWeights * RoutedExperts(x))
    """
    def __init__(self, config):
        super().__init__()
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.top_k = config.top_k_experts
        self.n_embd = config.n_embd
        
        # Router: decides which routed experts to use
        self.gate = nn.Linear(config.n_embd, config.n_routed_experts, bias=False)
        
        # Routed experts (selected dynamically)
        self.routed_experts = nn.ModuleList([
            Expert(config) for _ in range(config.n_routed_experts)
        ])
        
        # Shared experts (always active)
        self.shared_experts = nn.ModuleList([
            Expert(config) for _ in range(config.n_shared_experts)
        ]) if config.n_shared_experts > 0 else None
        
        # DeepSeek V3: Learnable bias for load balancing
        # This bias term adjusts routing scores to encourage balanced usage
        self.expert_bias = nn.Parameter(torch.zeros(config.n_routed_experts))
        
        # Running statistics for bias computation (not trained, just tracked)
        self.register_buffer('expert_usage_count', torch.zeros(config.n_routed_experts))
        self.register_buffer('total_tokens_processed', torch.tensor(0.0))
        
        # Bias update parameters
        self.bias_momentum = 0.99  # Momentum for updating bias
        self.target_usage = 1.0 / config.n_routed_experts  # Target uniform distribution
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, n_embd)
        Returns:
            output: (batch, seq_len, n_embd)
            aux_loss: None (loss-less load balancing)
        """
        B, T, C = x.shape
        
        # Flatten batch and sequence for routing
        x_flat = x.view(-1, C)  # (B*T, C)
        
        # Process shared experts (always active)
        shared_output = torch.zeros_like(x_flat)
        if self.shared_experts is not None:
            for shared_expert in self.shared_experts:
                shared_output += shared_expert(x_flat)
            # Average shared experts output
            shared_output = shared_output / len(self.shared_experts)
        
        # Compute router logits for routed experts
        router_logits = self.gate(x_flat)  # (B*T, n_routed_experts)
        
        # DeepSeek V3: Apply learned bias for load balancing
        # The bias adjusts logits to encourage balanced expert usage
        router_logits = router_logits + self.expert_bias.unsqueeze(0)
        
        # Get top-k experts per token
        router_probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Normalize probabilities of selected experts
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        # Update expert usage statistics (for bias adjustment)
        if self.training:
            self._update_expert_statistics(topk_indices)
        
        # Initialize routed output
        routed_output = torch.zeros_like(x_flat)
        
        # Route to top-k experts
        for i in range(self.top_k):
            expert_idx = topk_indices[:, i]
            expert_prob = topk_probs[:, i:i+1]
            
            # Process each expert
            for expert_id in range(self.n_routed_experts):
                mask = (expert_idx == expert_id)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.routed_experts[expert_id](expert_input)
                    routed_output[mask] += expert_prob[mask] * expert_output
        
        # Combine shared and routed outputs
        output = shared_output + routed_output
        
        # Reshape back
        output = output.view(B, T, C)
        
        # No auxiliary loss! Return None
        return output, None
    
    def _update_expert_statistics(self, topk_indices):
        """
        Update expert usage statistics and adjust bias accordingly.
        
        DeepSeek V3 approach:
        - Track how often each expert is used
        - Adjust bias to push under-used experts up and over-used experts down
        - This naturally balances expert usage without auxiliary loss
        """
        with torch.no_grad():
            # Count expert usage in this batch
            batch_usage = torch.zeros(self.n_routed_experts, device=topk_indices.device)
            for expert_id in range(self.n_routed_experts):
                batch_usage[expert_id] = (topk_indices == expert_id).float().sum()
            
            # Update running statistics with momentum
            batch_size = topk_indices.numel()
            self.expert_usage_count.mul_(self.bias_momentum).add_(
                batch_usage, alpha=(1 - self.bias_momentum)
            )
            self.total_tokens_processed.mul_(self.bias_momentum).add_(
                batch_size, alpha=(1 - self.bias_momentum)
            )
            
            # Compute current expert usage rates
            current_usage = self.expert_usage_count / (self.total_tokens_processed + 1e-8)
            
            # Adjust bias: increase bias for under-used experts, decrease for over-used
            # This encourages the model to naturally balance expert usage
            usage_diff = self.target_usage - current_usage
            
            # Update expert bias (small learning rate for stability)
            bias_lr = 0.001
            self.expert_bias.add_(usage_diff * bias_lr)


class DeepSeekBlock(nn.Module):
    """
    DeepSeek Transformer Block with MLA and MoE
    """
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.self_attn = MultiHeadLatentAttention(config)
        self.post_attention_layernorm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.mlp = MixtureOfExperts(config)

    def forward(self, x):
        # Attention block
        x = x + self.self_attn(self.input_layernorm(x))
        
        # MoE block
        residual = x
        x = self.post_attention_layernorm(x)
        x_out, _ = self.mlp(x)  # aux_loss is None with loss-less balancing
        x = residual + x_out
        
        return x


@dataclass
class DeepSeekConfig:
    """
    Configuration for DeepSeek model with MLA and MoE
    
    Key parameters:
    - compression_ratio: KV compression factor (kv_dim = n_embd // compression_ratio)
    - n_routed_experts: Number of experts that are dynamically selected
    - n_shared_experts: Number of experts that are always active
    - top_k_experts: Number of routed experts activated per token
    
    Note: Uses DeepSeek V3's loss-less load balancing (no aux_loss_coef needed)
    """
    vocab_size: int = 49152
    n_layer: int = 30
    n_head: int = 9
    n_embd: int = 576
    head_dim: int = 64
    intermediate_size: int = 1536
    max_seq_length: int = 512
    
    # MLA parameters
    compression_ratio: int = 8  # kv_lora_rank = n_embd // compression_ratio = 72
    
    # MoE parameters
    n_routed_experts: int = 8        # Routed experts
    n_shared_experts: int = 1        # Shared experts (always active)
    top_k_experts: int = 2           # Active routed experts per token
    
    # Other parameters
    rms_norm_eps: float = 1e-5
    rope_theta: float = 100000.0
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True


class DeepSeek(nn.Module):
    """
    DeepSeek Language Model with Multi-Head Latent Attention and Mixture of Experts
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.model = nn.ModuleDict(dict(
            embed_tokens = nn.Embedding(config.vocab_size, config.n_embd),
            layers = nn.ModuleList([DeepSeekBlock(config) for _ in range(config.n_layer)]),
            norm = RMSNorm(config.n_embd, eps=config.rms_norm_eps),
        ))
        
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.max_seq_length, \
            f"Cannot forward sequence of length {T}, max sequence length is only {self.config.max_seq_length}"
        
        x = self.model.embed_tokens(idx)
        
        # Pass through all transformer blocks
        # No auxiliary loss accumulation with loss-less load balancing
        for block in self.model.layers:
            x = block(x)
        
        x = self.model.norm(x)
        
        if self.config.tie_word_embeddings:
            logits = F.linear(x, self.model.embed_tokens.weight)
        else:
            logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # Only language modeling loss (no auxiliary loss)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model flops utilization (MFU)"""
        N = self.count_parameters()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.max_seq_length
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops
        mfu = flops_achieved / flops_promised
        return mfu
    
    def get_expert_statistics(self):
        """
        Get expert usage statistics from all MoE layers.
        Returns dict with usage info for monitoring load balancing.
        """
        stats = {}
        for layer_idx, block in enumerate(self.model.layers):
            moe = block.mlp
            
            # Get expert usage counts and bias
            usage = moe.expert_usage_count.cpu().numpy()
            bias = moe.expert_bias.detach().cpu().numpy()
            total = moe.total_tokens_processed.item()
            
            if total > 0:
                usage_pct = (usage / total) * 100
            else:
                usage_pct = usage * 0
            
            stats[f'layer_{layer_idx}'] = {
                'usage_percentage': usage_pct,
                'bias': bias,
                'total_tokens': total
            }
        
        return stats

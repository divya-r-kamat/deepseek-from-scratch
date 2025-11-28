# Training DeepSeek Model from Scratch
A 767M parameter DeepSeek architecture implementation trained for 10,000 steps on Shakespeare text,  Multi-Head Latent Attention (MHLA) and Mixture of Experts (MoE) with shared experts.

## Dataset
The Shakespeare Dataset : We train on Shakespeare’s complete works (input.txt ≈ 1.1MB), including all plays, sonnets, and poems..

    File size: ~1.1 MB
    
      class DataLoaderLite:
          def __init__(self, B, T):
              self.B = B  # Batch size
              self.T = T  # Sequence length
              
              with open('input.txt', 'r') as f:
                  text = f.read()
              
              tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
              tokens = tokenizer.encode(text)
              self.tokens = torch.tensor(tokens, dtype=torch.long)
              
## Dataset Statistics

    ======================================================================
    DATASET DIAGNOSTICS
    ======================================================================
    Total characters: 1,115,394
    Total tokens: 338,025
    Unique tokens in dataset: 11,706 / 50257 total vocab

     First 300 characters of data:
    ----------------------------------------------------------------------
    First Citizen:
    Before we proceed any further, hear me speak.
    
    All:
    Speak, speak.
    
    First Citizen:
    You are all resolved rather to die than to famish?
    
    All:
    Resolved. resolved.
    
    First Citizen:
    First, you know Caius Marcius is chief enemy to the people.
    
    All:
    We know't, we know't.
    
    First Citizen:
    Let us
    ----------------------------------------------------------------------
    ======================================================================
    
## Model Architecture

    DeepSeek(
      (model): ModuleDict(
        (embed_tokens): Embedding(49152, 576)
        (layers): ModuleList(
          (0-29): 30 x DeepSeekBlock(
            (input_layernorm): RMSNorm()
            (self_attn): MultiHeadLatentAttention(
              (q_proj): Linear(in_features=576, out_features=576, bias=False)
              (kv_a_proj_with_mqa): Linear(in_features=576, out_features=104, bias=False)
              (kv_b_proj): Linear(in_features=72, out_features=288, bias=False)
              (o_proj): Linear(in_features=576, out_features=576, bias=False)
              (attn_dropout): Dropout(p=0.0, inplace=False)
              (rope): RotaryPositionalEmbedding()
            )
            (post_attention_layernorm): RMSNorm()
            (mlp): MixtureOfExperts(
              (gate): Linear(in_features=576, out_features=8, bias=False)
              (routed_experts): ModuleList(
                (0-7): 8 x Expert(
                  (gate_proj): Linear(in_features=576, out_features=1536, bias=False)
                  (up_proj): Linear(in_features=576, out_features=1536, bias=False)
                  (down_proj): Linear(in_features=1536, out_features=576, bias=False)
                )
              )
              (shared_experts): ModuleList(
                (0): Expert(
                  (gate_proj): Linear(in_features=576, out_features=1536, bias=False)
                  (up_proj): Linear(in_features=576, out_features=1536, bias=False)
                  (down_proj): Linear(in_features=1536, out_features=576, bias=False)
                )
              )
            )
          )
        )
        (norm): RMSNorm()
      )
    )
## Model Configurations

    ======================================================================
    DeepSeek Model Configuration
    ======================================================================
    Model parameters: 767,447,088 (767.45M)
    Vocabulary size: 49,152
    Layers: 30
    Heads: 9
    Embedding dimension: 576
    Head dimension: 64
    Intermediate size: 1536
    Max sequence length: 512
    
    MLA Configuration:
      Compression ratio: 8
      KV latent rank: 72
      RoPE dimension: 32
    
    MoE Configuration:
      Total experts: 9 (8 routed + 1 shared)
      Top-K routing: 2
      Load balancing: DeepSeek V3 Loss-less (bias-based)
    ======================================================================

## Training

    Tokenizer vocabulary size: 49152
    Model vocabulary size: 49152
    Loaded 341094 valid tokens
    Token range: [17, 49130]
    ~333 batches per full pass
    Training Configuration:
      Total steps: 0 → 10000
      LR schedule: enabled (max=0.0003, min=2.9999999999999997e-05)
      Warmup steps: 100
      Batch size: 4
      Sequence length: 256
      Log interval: 100


### Training Results

    !python train1.py --steps 10000 --log-interval 100

    step     0 | loss 10.8919 | lr 0.000003 | tok/s    532.9 | MFU 1.90%
    step   100 | loss 5.5902 | lr 0.000300 | tok/s    900.1 | MFU 3.21%
    step   200 | loss 4.9802 | lr 0.000300 | tok/s    894.6 | MFU 3.19%
    step   300 | loss 4.7597 | lr 0.000300 | tok/s    901.8 | MFU 3.21%
    step   400 | loss 4.3575 | lr 0.000299 | tok/s    912.0 | MFU 3.25%
    step   500 | loss 4.7038 | lr 0.000299 | tok/s    913.3 | MFU 3.26%
    step   600 | loss 4.7575 | lr 0.000298 | tok/s    908.4 | MFU 3.24%
    step   700 | loss 4.3416 | lr 0.000298 | tok/s    907.5 | MFU 3.23%
    step   800 | loss 4.8049 | lr 0.000297 | tok/s    901.0 | MFU 3.21%
    step   900 | loss 4.6933 | lr 0.000296 | tok/s    895.2 | MFU 3.19%
    step  1000 | loss 4.5267 | lr 0.000295 | tok/s    895.6 | MFU 3.19%
    
    ======================================================================
    Expert Usage Statistics at Step 1000
    ======================================================================
    
    Layer 0:
      Expert usage: [16.079144  12.27481   11.436836  12.640474   6.4723186 12.652233
     15.579228  12.865043 ]
      Expert bias: [ 0.00617744 -0.01244683  0.00957891  0.01109656  0.01131915 -0.02972297
     -0.01450561  0.00803879]
    
    Layer 29:
      Expert usage: [ 6.9281416  5.4910154  8.6513195 14.362112   8.857218   2.0197287
     47.45937    6.231196 ]
      Expert bias: [ 0.04879386  0.08220264  0.04783177 -0.12712431  0.0374205   0.05978883
     -0.27900362  0.08294959]
    ======================================================================
    
    
    Saved checkpoint: deepseek_checkpoint_step_1000.pt
    step  1100 | loss 4.0251 | lr 0.000293 | tok/s    912.3 | MFU 3.25%
    step  1200 | loss 3.5524 | lr 0.000292 | tok/s    901.5 | MFU 3.21%
    step  1300 | loss 3.7390 | lr 0.000290 | tok/s    904.4 | MFU 3.22%
    step  1400 | loss 3.0675 | lr 0.000289 | tok/s    910.8 | MFU 3.25%
    step  1500 | loss 3.8719 | lr 0.000287 | tok/s    890.9 | MFU 3.18%
    step  1600 | loss 3.9138 | lr 0.000285 | tok/s    887.7 | MFU 3.16%
    step  1700 | loss 3.1028 | lr 0.000283 | tok/s    894.6 | MFU 3.19%
    step  1800 | loss 3.5746 | lr 0.000281 | tok/s    912.7 | MFU 3.25%
    step  1900 | loss 3.3691 | lr 0.000279 | tok/s    896.7 | MFU 3.20%
    step  2000 | loss 3.8258 | lr 0.000276 | tok/s    890.5 | MFU 3.17%
    
    ======================================================================
    Expert Usage Statistics at Step 2000
    ======================================================================
    
    Layer 0:
      Expert usage: [15.073069   7.2140446 11.168329  14.463268   7.0784082 13.941762
      8.141853  22.91968  ]
      Expert bias: [ 0.009981    0.0150505  -0.00720163 -0.03088293  0.05164025 -0.03402061
     -0.00448268 -0.0025813 ]
    
    Layer 29:
      Expert usage: [ 8.668771   7.3397427  9.458216  13.417835   8.747023   3.8409307
     40.65418    7.873706 ]
      Expert bias: [ 0.08404633  0.12666689  0.05941742 -0.16850111  0.06372597  0.11733868
     -0.56489795  0.1331392 ]
    ======================================================================
    
    
    Saved checkpoint: deepseek_checkpoint_step_2000.pt
    step  2100 | loss 3.1532 | lr 0.000274 | tok/s    896.0 | MFU 3.19%
    step  2200 | loss 2.7474 | lr 0.000271 | tok/s    890.2 | MFU 3.17%
    step  2300 | loss 2.9973 | lr 0.000268 | tok/s    905.6 | MFU 3.23%
    step  2400 | loss 2.4975 | lr 0.000266 | tok/s    784.3 | MFU 2.80%
    step  2500 | loss 2.6136 | lr 0.000263 | tok/s    902.9 | MFU 3.22%
    step  2600 | loss 2.4939 | lr 0.000260 | tok/s    896.9 | MFU 3.20%
    step  2700 | loss 2.5429 | lr 0.000257 | tok/s    903.8 | MFU 3.22%
    step  2800 | loss 2.1193 | lr 0.000253 | tok/s    903.3 | MFU 3.22%
    step  2900 | loss 2.2797 | lr 0.000250 | tok/s    908.7 | MFU 3.24%
    step  3000 | loss 2.3760 | lr 0.000247 | tok/s    897.6 | MFU 3.20%
    
    ======================================================================
    Expert Usage Statistics at Step 3000
    ======================================================================
    
    Layer 0:
      Expert usage: [12.754777  11.059637  10.2146635 16.401218  17.059105  15.437923
      5.7977858 11.275287 ]
      Expert bias: [ 0.00490673  0.03379495 -0.01371088 -0.06701437  0.09550018 -0.04825202
      0.04490575 -0.06046205]
    
    Layer 29:
      Expert usage: [11.960136   7.6564684 10.503282  14.132503   9.807469   4.3109574
     31.285572  10.344011 ]
      Expert bias: [ 0.10630667  0.17329182  0.05598044 -0.19671899  0.08806039  0.16072981
     -0.7566482   0.15994634]
    ======================================================================
    
    
    Saved checkpoint: deepseek_checkpoint_step_3000.pt
    step  3100 | loss 2.4026 | lr 0.000243 | tok/s    901.7 | MFU 3.21%
    step  3200 | loss 1.6834 | lr 0.000240 | tok/s    896.7 | MFU 3.20%
    step  3300 | loss 2.1632 | lr 0.000236 | tok/s    899.8 | MFU 3.21%
    step  3400 | loss 1.4888 | lr 0.000232 | tok/s    899.8 | MFU 3.21%
    step  3500 | loss 1.6332 | lr 0.000229 | tok/s    898.8 | MFU 3.20%
    step  3600 | loss 1.6558 | lr 0.000225 | tok/s    907.1 | MFU 3.23%
    step  3700 | loss 1.5221 | lr 0.000221 | tok/s    888.6 | MFU 3.17%
    step  3800 | loss 1.6193 | lr 0.000217 | tok/s    894.0 | MFU 3.19%
    step  3900 | loss 1.4947 | lr 0.000213 | tok/s    894.6 | MFU 3.19%
    step  4000 | loss 1.3407 | lr 0.000209 | tok/s    903.5 | MFU 3.22%
    
    ======================================================================
    Expert Usage Statistics at Step 4000
    ======================================================================
    
    Layer 0:
      Expert usage: [14.713444  19.54663    5.232686  13.258643  13.041317  23.624357
      2.2767105  8.306599 ]
      Expert bias: [-0.04849724  0.03051141  0.02032655 -0.08730104  0.11246303 -0.06533165
      0.09566269 -0.07796016]
    
    Layer 29:
      Expert usage: [12.061804   8.541582  12.06694   13.126092  10.131166   6.2250547
     26.645569  11.202206 ]
      Expert bias: [ 0.12203449  0.20861779  0.03786125 -0.22389588  0.10795157  0.20892327
     -0.87105775  0.16376688]
    ======================================================================
    
    
    Saved checkpoint: deepseek_checkpoint_step_4000.pt
    step  4100 | loss 1.0149 | lr 0.000205 | tok/s    902.4 | MFU 3.22%
    step  4200 | loss 0.9382 | lr 0.000201 | tok/s    902.4 | MFU 3.22%
    step  4300 | loss 1.0544 | lr 0.000197 | tok/s    890.4 | MFU 3.17%
    step  4400 | loss 0.9258 | lr 0.000193 | tok/s    908.8 | MFU 3.24%
    step  4500 | loss 0.9457 | lr 0.000188 | tok/s    900.5 | MFU 3.21%
    step  4600 | loss 1.0344 | lr 0.000184 | tok/s    907.3 | MFU 3.23%
    step  4700 | loss 0.7766 | lr 0.000180 | tok/s    907.4 | MFU 3.23%
    step  4800 | loss 1.0485 | lr 0.000176 | tok/s    904.6 | MFU 3.22%
    step  4900 | loss 0.7904 | lr 0.000171 | tok/s    901.0 | MFU 3.21%
    step  5000 | loss 0.8194 | lr 0.000167 | tok/s    903.0 | MFU 3.22%
    
    ======================================================================
    Expert Usage Statistics at Step 5000
    ======================================================================
    
    Layer 0:
      Expert usage: [14.331357  22.212011   4.3163357 13.451809  10.844982  20.05945
      6.281451   8.503004 ]
      Expert bias: [-0.09138392  0.00981046  0.0764399  -0.10314622  0.1149011  -0.10213833
      0.16404688 -0.09536964]
    
    Layer 29:
      Expert usage: [12.413084  8.676462 12.91123  13.20231  10.409238  6.805441 24.294306
     11.288328]
      Expert bias: [ 0.1313327   0.2439218   0.02115849 -0.24225356  0.12266496  0.24498764
     -0.96722025  0.17297602]
    ======================================================================
    
    
    Saved checkpoint: deepseek_checkpoint_step_5000.pt
    step  5100 | loss 0.7060 | lr 0.000163 | tok/s    882.8 | MFU 3.15%
    step  5200 | loss 0.5521 | lr 0.000159 | tok/s    892.0 | MFU 3.18%
    step  5300 | loss 0.7517 | lr 0.000154 | tok/s    888.7 | MFU 3.17%
    step  5400 | loss 0.4673 | lr 0.000150 | tok/s    887.2 | MFU 3.16%
    step  5500 | loss 0.6163 | lr 0.000146 | tok/s    881.6 | MFU 3.14%
    step  5600 | loss 0.5556 | lr 0.000142 | tok/s    880.7 | MFU 3.14%
    step  5700 | loss 0.5968 | lr 0.000137 | tok/s    892.5 | MFU 3.18%
    step  5800 | loss 0.5228 | lr 0.000133 | tok/s    899.7 | MFU 3.21%
    step  5900 | loss 0.7451 | lr 0.000129 | tok/s    869.4 | MFU 3.10%
    step  6000 | loss 0.5293 | lr 0.000125 | tok/s    895.0 | MFU 3.19%
    
    ======================================================================
    Expert Usage Statistics at Step 6000
    ======================================================================
    
    Layer 0:
      Expert usage: [21.69751   18.430038   3.698008  16.372124  11.555222  14.549719
      1.3181032 12.379665 ]
      Expert bias: [-0.15763237 -0.01821914  0.1420304  -0.11115921  0.10903439 -0.13884997
      0.24066673 -0.10120127]
    
    Layer 29:
      Expert usage: [12.337375  9.168473 12.376598 13.330473 11.642274  7.342585 22.500437
     11.302169]
      Expert bias: [ 0.13580114  0.27881366  0.00879642 -0.25863528  0.13732561  0.28155243
     -1.0544813   0.18063185]
    ======================================================================
    
    
    Saved checkpoint: deepseek_checkpoint_step_6000.pt
    step  6100 | loss 0.5559 | lr 0.000121 | tok/s    913.3 | MFU 3.26%
    step  6200 | loss 0.3938 | lr 0.000117 | tok/s    907.2 | MFU 3.23%
    step  6300 | loss 0.3480 | lr 0.000113 | tok/s    907.4 | MFU 3.23%
    step  6400 | loss 0.3886 | lr 0.000109 | tok/s    889.0 | MFU 3.17%
    step  6500 | loss 0.3962 | lr 0.000105 | tok/s    916.8 | MFU 3.27%
    step  6600 | loss 0.4969 | lr 0.000101 | tok/s    900.9 | MFU 3.21%
    step  6700 | loss 0.2849 | lr 0.000098 | tok/s    890.2 | MFU 3.17%
    step  6800 | loss 0.3543 | lr 0.000094 | tok/s    900.5 | MFU 3.21%
    step  6900 | loss 0.4032 | lr 0.000090 | tok/s    909.6 | MFU 3.24%
    step  7000 | loss 0.3524 | lr 0.000087 | tok/s    905.7 | MFU 3.23%
    
    ======================================================================
    Expert Usage Statistics at Step 7000
    ======================================================================
    
    Layer 0:
      Expert usage: [24.516544  18.619041   3.9926033 17.282816  12.441828   9.212974
      1.3209583 12.613633 ]
      Expert bias: [-0.2493562  -0.04656422  0.20553504 -0.13753852  0.09074973 -0.14538214
      0.3347277  -0.09725877]
    
    Layer 29:
      Expert usage: [12.482958   9.3941145 12.194753  12.563489  11.685807   8.013962
     21.693518  11.97178  ]
      Expert bias: [ 1.4212203e-01  3.1221238e-01  9.8343228e-04 -2.7424428e-01
      1.4587733e-01  3.1572247e-01 -1.1353745e+00  1.9312391e-01]
    ======================================================================
    
    
    Saved checkpoint: deepseek_checkpoint_step_7000.pt
    step  7100 | loss 0.3456 | lr 0.000083 | tok/s    895.5 | MFU 3.19%
    step  7200 | loss 0.2939 | lr 0.000080 | tok/s    889.2 | MFU 3.17%
    step  7300 | loss 0.2589 | lr 0.000077 | tok/s    902.0 | MFU 3.21%
    step  7400 | loss 0.2449 | lr 0.000073 | tok/s    904.9 | MFU 3.23%
    step  7500 | loss 0.2195 | lr 0.000070 | tok/s    901.0 | MFU 3.21%
    step  7600 | loss 0.2456 | lr 0.000067 | tok/s    902.0 | MFU 3.21%
    step  7700 | loss 0.1764 | lr 0.000064 | tok/s    898.7 | MFU 3.20%
    step  7800 | loss 0.1598 | lr 0.000062 | tok/s    903.2 | MFU 3.22%
    step  7900 | loss 0.2926 | lr 0.000059 | tok/s    896.6 | MFU 3.20%
    step  8000 | loss 0.1881 | lr 0.000056 | tok/s    900.8 | MFU 3.21%
    
    ======================================================================
    Expert Usage Statistics at Step 8000
    ======================================================================
    
    Layer 0:
      Expert usage: [26.392937  16.598173   5.3842363 16.851141   9.388752  12.416204
      2.3025866 10.666334 ]
      Expert bias: [-0.3638916  -0.07583446  0.2745223  -0.16604394  0.0750634  -0.12645648
      0.43204427 -0.09771246]
    
    Layer 29:
      Expert usage: [12.449775   9.749398  11.940994  12.558274  11.706364   8.6534
     21.134476  11.8077135]
      Expert bias: [ 0.14452063  0.34521002 -0.00250393 -0.28699034  0.15564463  0.34883177
     -1.211752    0.20189029]
    ======================================================================
    
    
    Saved checkpoint: deepseek_checkpoint_step_8000.pt
    step  8100 | loss 0.1603 | lr 0.000054 | tok/s    907.5 | MFU 3.23%
    step  8200 | loss 0.1322 | lr 0.000051 | tok/s    907.5 | MFU 3.23%
    step  8300 | loss 0.2252 | lr 0.000049 | tok/s    908.5 | MFU 3.24%
    step  8400 | loss 0.1562 | lr 0.000047 | tok/s    887.0 | MFU 3.16%
    step  8500 | loss 0.1434 | lr 0.000045 | tok/s    890.1 | MFU 3.17%
    step  8600 | loss 0.0668 | lr 0.000043 | tok/s    908.6 | MFU 3.24%
    step  8700 | loss 0.0595 | lr 0.000041 | tok/s    888.7 | MFU 3.17%
    step  8800 | loss 0.0792 | lr 0.000040 | tok/s    884.6 | MFU 3.15%
    step  8900 | loss 0.0784 | lr 0.000038 | tok/s    905.3 | MFU 3.23%
    step  9000 | loss 0.0987 | lr 0.000037 | tok/s    890.3 | MFU 3.17%
    
    ======================================================================
    Expert Usage Statistics at Step 9000
    ======================================================================
    
    Layer 0:
      Expert usage: [29.228365   7.745728   7.8434186 25.614422   3.2310538  6.8875976
      2.7659624 16.683846 ]
      Expert bias: [-0.5177323  -0.06949298  0.33156034 -0.24051966  0.13201055 -0.10744079
      0.52505267 -0.10255808]
    
    Layer 29:
      Expert usage: [12.160991  9.959645 12.575486 11.968757 11.868493  9.386153 19.591171
     12.489691]
      Expert bias: [ 0.14593655  0.3760315  -0.00395801 -0.2992901   0.16063364  0.37904364
     -1.2806187   0.2111744 ]
    ======================================================================
    
    
    Saved checkpoint: deepseek_checkpoint_step_9000.pt
    step  9100 | loss 0.0951 | lr 0.000035 | tok/s    895.6 | MFU 3.19%
    step  9200 | loss 0.0538 | lr 0.000034 | tok/s    902.4 | MFU 3.22%
    step  9300 | loss 0.0930 | lr 0.000033 | tok/s    912.4 | MFU 3.25%
    step  9400 | loss 0.0549 | lr 0.000032 | tok/s    900.9 | MFU 3.21%
    step  9500 | loss 0.0453 | lr 0.000032 | tok/s    901.1 | MFU 3.21%
    step  9600 | loss 0.0685 | lr 0.000031 | tok/s    891.2 | MFU 3.18%
    step  9700 | loss 0.0524 | lr 0.000031 | tok/s    874.0 | MFU 3.12%
    step  9800 | loss 0.0659 | lr 0.000030 | tok/s    898.6 | MFU 3.20%
    step  9900 | loss 0.0647 | lr 0.000030 | tok/s    901.6 | MFU 3.21%
    
    ======================================================================
    Training Complete!
    Final loss: 0.0526
    ======================================================================

## Quantization

Since the size of the model is very big quantized the model to 8bit.


    import torch
    
    src = '/content/deepseek-from-scratch/deepseek_checkpoint.pt'
    dst = '/content/deepseek-from-scratch/deepseek_checkpoint_int8.pt'
    
    ckpt = torch.load(src, map_location='cpu', weights_only=False)
    sd = ckpt['model_state']
    
    quant_state = {}
    meta = {}  # store scales and dtypes for dequant
    
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            # Keep non-tensor entries as-is
            quant_state[k] = v
            meta[k] = {'type': 'non_tensor'}
            continue
    
        if v.is_floating_point():
            # Symmetric per-tensor int8 quantization
            vmax = v.abs().max()
            # Avoid division by zero
            scale = (vmax / 127.0).item() if vmax > 0 else 1e-8
    
            q = torch.clamp((v / scale).round(), -127, 127).to(torch.int8)
            quant_state[k] = q
            meta[k] = {'type': 'int8', 'scale': scale, 'orig_dtype': str(v.dtype)}
        else:
            # Non-float tensors (e.g., embeddings indices, buffers) kept as-is
            quant_state[k] = v
            meta[k] = {'type': 'raw', 'orig_dtype': str(v.dtype)}
    
    torch.save({
        'quant_state': quant_state,
        'meta': meta,
        'config': ckpt.get('config', None),
        'loss': ckpt.get('loss', None),
        'step': ckpt.get('step', None),
    }, dst)
    
    print(f"Saved INT8 checkpoint to: {dst}")


## Output Sample

    ================================================================================
    DeepSeek Text Generation from Hugging Face
    ================================================================================
    Using device: cuda
    
    Downloading model from Hugging Face...
    Repository: dkamat/deepseek-from-stratch
    File: deepseek_model.pt
    deepseek_model.pt: 100% 776M/776M [00:07<00:00, 97.1MB/s]
    ✓ Model downloaded successfully!
    Location: ./model_cache/spaces--dkamat--deepseek-from-stratch/snapshots/8ca35236a2b3e0a2babc963d7e94b8b56da4d248/deepseek_model.pt
    
    Loading quantized checkpoint from ./model_cache/spaces--dkamat--deepseek-from-stratch/snapshots/8ca35236a2b3e0a2babc963d7e94b8b56da4d248/deepseek_model.pt...
    Dequantizing model weights...
    ✓ Dequantization complete!
    
    Initializing model...
    ✓ Model loaded successfully!
    Training step: 10000
    Training loss: 0.0526
    Vocabulary size: 49152
    
    Loading tokenizer...
    tokenizer_config.json: 3.66kB [00:00, 7.01MB/s]
    vocab.json: 801kB [00:00, 120MB/s]
    merges.txt: 466kB [00:00, 150MB/s]
    tokenizer.json: 2.10MB [00:00, 207MB/s]
    special_tokens_map.json: 100% 831/831 [00:00<00:00, 7.66MB/s]
    ✓ Tokenizer loaded
    
    Prompt: 'To thine own self'
    
    Generating 5 samples...
    Parameters: max_tokens=200, temp=0.7, top_k=200, top_p=0.9
    ================================================================================
    
    ================================================================================
    SAMPLE 1/5
    ================================================================================
    To thine own selfsame words?
    
    CORIOLANUS:
    What is't o'clock?
    
    MARCIUS:
    You are sent for to the senate: but for the field
    We have lived to do so.
    
    SICINIUS:
    Why, 'tis no great matter; for the state hath
    delivered the common people of your braggart, you
    have not many things as you.
    
    SICINIUS:
    Have you a cup of pride now, you have shamed a
    butter thus to a block-place; to the remove
    afflicts your asking: take your voices; and your minds
    you are not such as enemies. You are plebeians,
    and make a petty servant to choose an hour
     how to take and one crutch; that's in me, I
    if it were at liberty, ever spake against
    him last night. You talk of pride: O, I am
    
    ================================================================================
    
    ================================================================================
    SAMPLE 2/5
    ================================================================================
    To thine own selfsame words?
    
    CORIOLANUS:
    Nay, let them know more.
    
    CORIOLANUS:
    We'll be sworn they are too rough.
    The gods preserve is open that our first day,
    The dust is given already.
    
    MARCIUS:
    You are sent for to the people, what you have
    Desire to be so valiant.
    
    MENENIUS:
    Hear me, sir.
    
    AEdile:
    No, no, nor no, away!
    
    CORIOLANUS:
    Bid them wash him
    The grave of our seats and envy
    With rigorous that would lie unsweptial.
    
    SICINIUS:
    Flower of warriors,
    How is it with Titus Lartius?
    
    MARCIUS:
    Nay, let them follow:
    What cannot be much, if you be yours,
    Thou hast
    
    ================================================================================
    
    ================================================================================
    SAMPLE 3/5
    ================================================================================
    To thine own selfsame words?
    
    CORIOLANUS:
    Nay, sit down;
    Beseech you, hear me no less than his country.
    
    CORIOLANUS:
    Why, so he did, I am sure.
    
    Citizens:
    No, no, no more.
    
    Second Citizen:
    Not unlike, to see him he is a drop of your
    trueceiving back on a pleasing tongue of allaying
    Tiber in't; O, a strange one hour, his wounds!
    
    First Citizen:
    He cannot but with measure fit the honours
    Which he so sets at nought.
    
    First Citizen:
    He's right noble:
    Go to, and see him he.
    
    SICINIUS:
    A crack, you are well understood to her.
    
    MENENIUS:
    A match, sir, your mother, so, that have found
    An o' the
    
    ================================================================================
    
    ================================================================================
    SAMPLE 4/5
    ================================================================================
    To thine own selfsame words?
    
    BRUTUS:
    Hang?
    
    VOLUMNIA:
    You have ever your gates the body in some more
    pleasure to speak: she is a single plot
    With no better nature, so that to say he would
    Have made to crush your voices?
    
    VOLUMNIA:
    Prithee, no, no,
    I would have had you truly: so, you shall know
    What you can do more than stay till it is
    Wish we'll beat Aufidius.
    
    MENENIUS:
    Do not hear their friends to see't--but, if
    'Tis, I'll potch you, hear me speak.
    
    First Senator:
    What, shall we do it in?
    
    VOLUMNIA:
    I shall be to do it,--
    
    SICINIUS:
    Pray you, mark me,--but
    
    ================================================================================
    
    ================================================================================
    SAMPLE 5/5
    ================================================================================
    To thine own selfsame words?
    
    COMINIUS:
    Ay, but not angry.
    
    CORIOLANUS:
    You have received many wounds, and cannot tell us.
    
    CORIOLANUS:
    We know neither me, if it be not too much,
    He could have one scratch my poor host freedom
    From whence I am. You, for, be gone:
    The gods give you joy and to be their bedfellow.
    
    GONZALO:
    Mere half to be a humorous patrician,
    Your honour.
    
    CORIOLANUS:
    Nay, these are almost thoroughly persuaded; or veil
    With cloth them, or pitying, bear
    My praises are too infant.
    
    SICINIUS:
    What then?
    
    CORIOLANUS:
    On safe-guard he came to me; and did curse
    Against the Volsces, for they had
    
    ================================================================================
    
    Saved checkpoint: deepseek_checkpoint.pt

"""
microgpt in PyTorch (GPU)，在 TinyStories 上训练，tokenizer 为 bpe1024，宽度 48，bf16 混合精度，打开 1.09 的全部加速。
对比训练细节：初始化、weight decay、梯度裁剪、dropout。每次只改一项，其余保持默认。
用法：python train.py [init=0.08] [wd=none] [clip=none] [dropout=0] [lr=0.01] [seed=42] [frac=1]     例如 init=xavier lr=0.005

Based on microgpt by @karpathy
"""

import os       # os.path.join
import sys      # sys.argv
import math     # math.log
import random   # random.seed, random.choices, random.gauss
import time     # time.perf_counter
import torch
import torch.nn.functional as F # F.scaled_dot_product_attention, F.dropout
args = dict(init='0.08', wd='none', clip='none', dropout='0', lr='0.01', seed='42', frac='1') # 默认值
args.update(a.split('=') for a in sys.argv[1:])
init     = args['init']     # 初始化：zero、xavier、kaiming、0.02、0.08、zero_proj
wd       = args['wd']       # weight decay：none、l2、adamw、adamw_noemb
clip     = args['clip']     # 梯度裁剪：none，或者梯度总长度的上限（如 1.0）
dropout  = args['dropout']  # dropout：0、0.1（全程）、0.1late（后半段才开）
learning_rate, seed = float(args['lr']), int(args['seed'])
num_steps = 4096
frac = int(args['frac']) # 只用训练集的前 1/frac，让模型能死记数据；验证集不变
random.seed(seed) # Let there be order among chaos
torch.manual_seed(seed)
autocast = lambda: torch.autocast('cuda', dtype=torch.bfloat16) # 1.08 已经对比过：bf16 混合精度
torch.set_num_threads(1)
device = 'cuda:0'

# 项目路径：从 data/tok/ 读数据，日志写到 out/<这一节>/init<init>_wd<wd>_clip<clip>_drop<dropout>_lr<lr>_s<seed>/
tokenizer = 'bpe1024'
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
run_name = f"init{init}_wd{wd}_clip{clip}_drop{dropout}_lr{learning_rate:g}_s{seed}" + (f"_f{frac}" if frac != 1 else "")
out_dir = os.path.join(root, 'out', os.path.basename(os.path.dirname(os.path.abspath(__file__))), run_name)
os.makedirs(out_dir, exist_ok=True)

# Let there be tokenized data: two streams of token ids, and the bytes each token stands for
data = torch.load(os.path.join(root, 'data', 'tok', f'{tokenizer}.pt'))
train_data, val_data, vocab, BOS = data['train'], data['val'], data['vocab'], data['bos']
train_data = train_data[:len(train_data) // frac]
vocab_size = len(vocab)
token_bytes = torch.tensor([len(b) for b in vocab], device=device) # to turn loss per token into bits per byte
decode = lambda ids: b''.join(vocab[i] for i in ids).decode('utf-8', errors='replace')
print(f"tokenizer: {tokenizer}, vocab size: {vocab_size}, train tokens: {len(train_data)}, val tokens: {len(val_data)}")

# Initialize the parameters, to store the knowledge of the model
n_layer = 1     # depth of the transformer neural network (number of layers)
n_embd = 48     # width of the network (embedding dimension)
block_size = 256 # maximum context length of the attention window
n_head = 4      # number of attention heads
head_dim = n_embd // n_head # derived dimension of each head
def init_std(name, nout, nin):
    # 每个矩阵从 N(0, std²) 里取值，std 怎么定见 README 第 1 节。nin 就是 fan_in（每个输出由多少个输入加起来），nout 是 fan_out
    if init == 'zero':
        return 0.0
    if init == 'xavier':
        return math.sqrt(2 / (nin + nout))
    if init == 'kaiming':
        return math.sqrt(2 / nin)
    if init == 'zero_proj': # 残差分支的输出层置零，其余 0.02
        return 0.0 if name.endswith(('attn_wo', 'mlp_fc2')) else 0.02
    return float(init) # 固定常数，如 0.02、0.08
matrix = lambda name, nout, nin: torch.tensor([[random.gauss(0, init_std(name, nout, nin)) for _ in range(nin)] for _ in range(nout)], device=device, requires_grad=True)
shapes = {'wte': (vocab_size, n_embd), 'wpe': (block_size, n_embd), 'lm_head': (vocab_size, n_embd)}
for i in range(n_layer):
    shapes[f'layer{i}.attn_wq'] = (n_embd, n_embd)
    shapes[f'layer{i}.attn_wk'] = (n_embd, n_embd)
    shapes[f'layer{i}.attn_wv'] = (n_embd, n_embd)
    shapes[f'layer{i}.attn_wo'] = (n_embd, n_embd)
    shapes[f'layer{i}.mlp_fc1'] = (4 * n_embd, n_embd)
    shapes[f'layer{i}.mlp_fc2'] = (n_embd, 4 * n_embd)
state_dict = {name: matrix(name, *shape) for name, shape in shapes.items()}
params = list(state_dict.values())
n_params = sum(p.numel() for p in params)
n_vocab_params = state_dict['wte'].numel() + state_dict['lm_head'].numel()
print(f"num params: {n_params} ({n_vocab_params} in wte and lm_head), init {init}: " + ', '.join(f"{k} {p.std().item():.3f}" for k, p in state_dict.items()))

# Define the model architecture: a function mapping tokens and parameters to logits over what comes next
# Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
def linear(x, w):
    return x @ w.T

def softmax(logits):
    exps = (logits - logits.max(dim=-1, keepdim=True).values).exp()
    return exps / exps.sum(dim=-1, keepdim=True)

def rmsnorm(x):
    return x * (x.pow(2).mean(dim=-1, keepdim=True) + 1e-5) ** -0.5

def gpt(idx, p_drop=0.0):
    # p_drop：dropout 的概率，只在训练时大于 0（GPT-2 的位置：embedding 之后、注意力权重上、每个子层的输出上）
    B, T = idx.shape
    tok_emb = state_dict['wte'][idx] # token embedding                        (B, T, C)
    pos_emb = state_dict['wpe'][:T] # position embedding                      (T, C)
    x = F.dropout(tok_emb + pos_emb, p_drop) # joint token and position embedding
    x = rmsnorm(x) # note: not redundant due to backward pass via the residual connection

    for li in range(n_layer):
        # 1) Multi-head Attention block, all positions and all heads at once
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq']).view(B, T, n_head, head_dim).transpose(1, 2) # (B, H, T, D)
        k = linear(x, state_dict[f'layer{li}.attn_wk']).view(B, T, n_head, head_dim).transpose(1, 2) # (B, H, T, D)
        v = linear(x, state_dict[f'layer{li}.attn_wv']).view(B, T, n_head, head_dim).transpose(1, 2) # (B, H, T, D)
        x_attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p_drop)          # (B, H, T, D)
        x_attn = x_attn.transpose(1, 2).reshape(B, T, n_embd)                                         # (B, T, C)
        x = F.dropout(linear(x_attn, state_dict[f'layer{li}.attn_wo']), p_drop)
        x = x + x_residual
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = x.relu()
        x = F.dropout(linear(x, state_dict[f'layer{li}.mlp_fc2']), p_drop)
        x = x + x_residual

    logits = linear(x, state_dict['lm_head']) # (B, T, vocab_size)
    return logits

def token_losses(x, y, p_drop=0.0):
    with autocast():
        probs = softmax(gpt(x, p_drop))
        return -probs.gather(-1, y.unsqueeze(-1)).squeeze(-1).log() # loss of every position, in nats

# A batch is batch_size random windows of block_size tokens; the target is the same window shifted by one
# 训练数据整份放在 GPU 上，只把 16 个起点传过去，在 GPU 上切出窗口（1.09）
def get_batch(data, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,)).pin_memory().to(device, non_blocking=True)
    offsets = ix[:, None] + torch.arange(block_size, device=device)
    return data[offsets], data[offsets + 1]

# Validation: the same 500 stories for every tokenizer, measured in bits per byte so tokenizers can be compared
val_x = torch.stack([val_data[i:i + block_size] for i in range(0, len(val_data) - block_size - 1, block_size)]).to(device)
val_y = torch.stack([val_data[i + 1:i + block_size + 1] for i in range(0, len(val_data) - block_size - 1, block_size)]).to(device)
# 训练集上也取同样多的固定窗口，用同样的方式评估，看训练集和验证集的差距（有没有死记训练数据）
train_x = torch.stack([train_data[i:i + block_size] for i in range(0, len(val_x) * block_size, block_size)]).to(device)
train_y = torch.stack([train_data[i + 1:i + block_size + 1] for i in range(0, len(val_x) * block_size, block_size)]).to(device)

@torch.no_grad()
def bpb(xs, ys):
    nats = sum(token_losses(xs[i:i + 8], ys[i:i + 8]).double().sum() for i in range(0, len(xs), 8)).item() # 求和用 fp64，否则累加误差很大
    return nats / math.log(2) / token_bytes[ys].sum().item()

# 优化器：Adam（1.06 已经对比过），fused 实现（1.09）。weight decay 的几种做法见 README 第 2 节
if wd == 'none':
    optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(0.85, 0.99), eps=1e-8, fused=True)
elif wd == 'l2': # 加在 loss 上：梯度里多一项 0.1·θ，再和梯度一起被 Adam 归一化
    optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(0.85, 0.99), eps=1e-8, weight_decay=0.1, fused=True)
elif wd == 'adamw': # 和梯度分开：每步单独减掉 lr·0.1·θ
    optimizer = torch.optim.AdamW(params, lr=learning_rate, betas=(0.85, 0.99), eps=1e-8, weight_decay=0.1, fused=True)
elif wd == 'adamw_noemb': # 同上，但 wte、wpe 不加（OLMo 2）
    groups = [{'params': [p for k, p in state_dict.items() if k in ('wte', 'wpe')], 'weight_decay': 0.0},
              {'params': [p for k, p in state_dict.items() if k not in ('wte', 'wpe')], 'weight_decay': 0.1}]
    optimizer = torch.optim.AdamW(groups, lr=learning_rate, betas=(0.85, 0.99), eps=1e-8, fused=True)
# 学习率调度：线性衰减到 0（1.07 已经对比过）
lr_mult = lambda step: 1 - step / num_steps
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_mult)
max_norm = float('inf') if clip == 'none' else float(clip) # 上限为无穷大时不裁剪，但仍然算出梯度的总长度，用来记录
print(f"run: {run_name}")

# 训练一步的前向：交给 torch.compile 编译，并用 CUDA Graph（1.09）。p_drop 换一个值会重新编译一次
@torch.compile(mode='reduce-overhead')
def batch_loss(x, y, p_drop):
    return token_losses(x, y, p_drop).mean()
def drop_at(step): # 这一步的 dropout 概率
    if dropout == '0.1late':
        return 0.1 if step >= num_steps // 2 else 0.0
    return float(dropout)

batch_size = 16
num_evals = 32
train_bytes_per_token = token_bytes[train_data.to(device)].double().mean().item()
print(f"bytes per token: {train_bytes_per_token:.2f}")
train_data = train_data.to(device) # 整份训练数据只有约 48 MB，直接放进显存

log = open(os.path.join(out_dir, 'log.csv'), 'w')
log.write('step,tokens,text_bytes,time,lr,train_loss,train_bpb,val_bpb,grad_norm_mean,grad_norm_max,clipped_frac\n')
log.write(f"0,0,0,0,{learning_rate * lr_mult(0):.6g},,{bpb(train_x, train_y):.6f},{bpb(val_x, val_y):.6f},,,\n")
# 计时：每 128 步评估一次，计时只算两次评估之间的训练；读时间前先等 GPU 把排队的活干完
eval_every = num_steps // num_evals
zero = lambda: torch.zeros((), device=device)
train_time, loss_sum, norm_sum, norm_max, n_clipped, diverged = 0.0, zero(), zero(), zero(), zero(), False
torch.cuda.synchronize()
t0 = time.perf_counter()
for step in range(num_steps):
    x, y = get_batch(train_data, batch_size)
    torch.compiler.cudagraph_mark_step_begin() # 告诉 CUDA Graph：新的一步开始了，上一步的输出可以覆盖
    loss = batch_loss(x, y, drop_at(step))
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(params, max_norm) # 返回裁剪前的总长度；超过 max_norm 时，所有梯度按比例缩小到 max_norm
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()

    # 都留在 GPU 上累加，不同步（1.09）
    loss_sum += loss.detach()
    norm_sum += norm
    norm_max = torch.maximum(norm_max, norm)
    n_clipped += (norm > max_norm).float()
    if (step + 1) % eval_every == 0:
        torch.cuda.synchronize()
        train_time += time.perf_counter() - t0
        train_loss = loss_sum.item() / eval_every
        if not math.isfinite(train_loss): # 学习率太大，训练发散了
            print(f"\ndiverged before step {step+1}")
            diverged = True
            break
        tokens = (step + 1) * batch_size * block_size
        train_bpb, val_bpb = bpb(train_x, train_y), bpb(val_x, val_y)
        log.write(f"{step+1},{tokens},{tokens * train_bytes_per_token:.0f},{train_time:.4f},{learning_rate * lr_mult(step):.6g},{train_loss:.6f},"
                  f"{train_bpb:.6f},{val_bpb:.6f},{norm_sum.item() / eval_every:.4f},{norm_max.item():.4f},{n_clipped.item() / eval_every:.4f}\n")
        print(f"step {step+1:5d} / {num_steps:5d} | train loss {train_loss:.4f} | val bpb {val_bpb:.4f} | grad norm {norm_sum.item() / eval_every:.3f}", end='\r')
        loss_sum, norm_sum, norm_max, n_clipped = zero(), zero(), zero(), zero()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
log.close()
if diverged:
    sys.exit()

print(f"\ntrain bpb {train_bpb:.4f}, val bpb {val_bpb:.4f}, time {train_time:.1f}s")
with open(os.path.join(out_dir, 'final.csv'), 'w') as f:
    f.write('train_bpb,val_bpb,time\n')
    f.write(f"{train_bpb:.6f},{val_bpb:.6f},{train_time:.4f}\n")

# Inference: may the model babble back to us
temperature = 0.5 # in (0, 1], control the "creativity" of generated text, low to high
samples = open(os.path.join(out_dir, 'samples.txt'), 'w')
print("\n--- inference (new, hallucinated stories) ---")
with torch.no_grad():
    for sample_idx in range(3):
        idx = [BOS]
        for _ in range(block_size - 1):
            logits = gpt(torch.tensor([idx], device=device))[0, -1]
            probs = softmax(logits / temperature)
            token_id = random.choices(range(vocab_size), weights=probs.tolist())[0]
            if token_id == BOS:
                break
            idx.append(token_id)
        sample = decode(idx[1:])
        print(f"sample {sample_idx+1}: {sample}\n")
        samples.write(sample + '\n\n')
samples.close()

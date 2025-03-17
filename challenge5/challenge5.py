import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda'

### Transformation Functions ###

def transformation_function_CE(batch, weight, labels):
    """
    Compute cross entropy loss in sum-reduction mode without FP32 upcast.
    Expects:
      - batch: (B, S, D)
      - weight: (vocab, D)
      - labels: (B, S) or (B*S,) for CE loss.
    """
    x = F.linear(batch, weight)  # no upcast; computed in bfloat16
    loss_fct = nn.CrossEntropyLoss(reduction="sum")
    loss = loss_fct(x.view(-1, x.size(-1)), labels.view(-1))
    return loss

def transformation_function_focal(batch, weight, labels, gamma=1.0, eps=1e-7):
    """
    Compute focal loss in sum-reduction mode.
    Expects:
      - batch: (B, S, D)
      - weight: (vocab, D)
      - labels: (B, S) or (B*S,) for classification.
    """
    x = F.linear(batch, weight)
    log_p = F.log_softmax(x, dim=-1)
    p = log_p.exp()
    target = labels.view(-1).long()
    p_t = p.view(-1, x.size(-1)).gather(1, target.unsqueeze(1)) + eps
    focal_factor = (1 - p_t) ** gamma
    loss = - (focal_factor * log_p.view(-1, x.size(-1)).gather(1, target.unsqueeze(1))).sum()
    return loss

### Memory-Efficient Linear Function ###

class MemoryEfficientLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, labels, forward_function, batch_chunk_size, seq_chunk_size):
        """
        X: (B, S, D)
        weight: (vocab, D)
        labels: for CE and focal: (B*S,) (will be reshaped to (B,S))
        forward_function: function to compute loss for a chunk.
        batch_chunk_size: number of examples (B) per chunk.
        seq_chunk_size: number of tokens (S) per chunk.
        """
        ctx.save_for_backward(X, weight, labels)
        ctx.batch_chunk_size = batch_chunk_size
        ctx.seq_chunk_size = seq_chunk_size
        ctx.forward_function = forward_function

        B, S, _ = X.shape
        
        # For CE, labels is 1D; for focal, also 1D.
        if labels.dim() == 1:
            labels_reshaped = labels.view(B, S)
        elif labels.dim() == 3:
            labels_reshaped = labels
        else:
            raise ValueError("Labels must be 1D or 3D.")

        # Accumulate loss in FP32 for better numerical precision.
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=X.device)
        total_tokens = 0
        for i in range(0, B, batch_chunk_size):
            X_batch = X[i:i+batch_chunk_size]
            labels_batch = labels_reshaped[i:i+batch_chunk_size]
            for j in range(0, S, seq_chunk_size):
                X_chunk = X_batch[:, j:j+seq_chunk_size]
                if labels_reshaped.dim() == 2:
                    labels_chunk = labels_batch[:, j:j+seq_chunk_size]
                else:
                    labels_chunk = labels_batch[:, j:j+seq_chunk_size, :]
                chunk_loss = forward_function(X_chunk, weight, labels_chunk)
                total_loss += chunk_loss.float()
                total_tokens += X_chunk.size(0) * X_chunk.size(1)
        ctx.total_tokens = total_tokens
        final_loss = total_loss / total_tokens if total_tokens != 0 else torch.tensor(0.0, device=X.device)
        # Return in the same dtype as X (bfloat16)
        return final_loss.to(X.dtype)

    @staticmethod
    def backward(ctx, d_loss):
        X, W, labels = ctx.saved_tensors
        batch_chunk_size = ctx.batch_chunk_size
        seq_chunk_size = ctx.seq_chunk_size
        forward_function = ctx.forward_function
        total_tokens = ctx.total_tokens
        B, S, _ = X.shape

        d_X = torch.zeros_like(X) if X.requires_grad else None
        d_W = torch.zeros_like(W) if W.requires_grad else None

        if labels.dim() == 1:
            labels_reshaped = labels.view(B, S)
        elif labels.dim() == 3:
            labels_reshaped = labels
        else:
            raise ValueError("Labels must be 1D or 3D.")

        for i in range(0, B, batch_chunk_size):
            X_batch = X[i:i+batch_chunk_size]
            labels_batch = labels_reshaped[i:i+batch_chunk_size]
            for j in range(0, S, seq_chunk_size):
                X_chunk = X_batch[:, j:j+seq_chunk_size].detach().requires_grad_(True)
                labels_chunk = labels_batch[:, j:j+seq_chunk_size]
                with torch.enable_grad():
                    chunk_loss = forward_function(X_chunk, W, labels_chunk)
                    # Use the same uniform scaling as in forward: divide by total_tokens.
                    local_loss = chunk_loss / total_tokens
                    gX, gW = torch.autograd.grad(local_loss, (X_chunk, W), retain_graph=True)
                if d_X is not None:
                    d_X[i:i+batch_chunk_size, j:j+seq_chunk_size] += gX * d_loss
                if d_W is not None:
                    d_W += gW * d_loss
        return d_X, d_W, None, None, None, None

### Reference Loss Functions ###

def reference_loss_fn_CE(X, W, labels):
    logits = F.linear(X, W)
    B, S, _ = X.shape
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="sum")
    return loss / (B * S)

def reference_loss_fn_focal(X, W, labels, gamma=1.0, eps=1e-7):
    logits = F.linear(X, W)
    log_p = F.log_softmax(logits, dim=-1)
    p = log_p.exp()
    target = labels.view(-1).long()
    p_t = p.view(-1, logits.size(-1)).gather(1, target.unsqueeze(1)) + eps
    focal_factor = (1 - p_t) ** gamma
    loss = - (focal_factor * log_p.view(-1, logits.size(-1)).gather(1, target.unsqueeze(1))).sum()
    B, S, _ = X.shape
    return loss / (B * S)

### Llama-1B Training Loss Validation ###

def validate_llama_training_loss_matches():
    """
    Validate that the memory-efficient linear function produces a training loss
    that matches the full (reference) computation under Llama-1B parameters.
    """
    bsz, qlen, hd, vocab = 4, 4096, 4096, 128000  # Llama-1B parameters
    X_large = torch.randn(bsz, qlen, hd, dtype=torch.bfloat16, device=device, requires_grad=True)
    W_large = torch.randn(vocab, hd, dtype=torch.bfloat16, device=device, requires_grad=True)
    labels_large = torch.randint(0, vocab, (bsz * qlen,), device=device)
    
    # Compute reference loss with sum reduction divided by (B*S)
    loss_ref = F.cross_entropy(F.linear(X_large, W_large).view(-1, vocab),
                               labels_large.view(-1),
                               reduction="sum") / (bsz * qlen)
    
    # Compute memory-efficient loss.
    loss_mem = MemoryEfficientLinear.apply(X_large, W_large, labels_large, transformation_function_CE, 1, 1024)
    
    print("Llama-1B Reference Loss: {:.4f}".format(loss_ref.item()))
    print("Llama-1B Memory-Efficient Loss: {:.4f}".format(loss_mem.item()))
    if torch.allclose(torch.tensor(loss_ref.item()), torch.tensor(loss_mem.item()), atol=1e-3):
        print("Llama-1B training loss matches!")
    else:
        print("Llama-1B training loss does NOT match.")

### Validation Routine for Other Functions ###

def validate_GRPO_memory_efficient_linear(loss_fn, transformation_fn, label_dim, loss_name=""):
    bsz, qlen, hd, vocab = 2, 1024, 512, 10000
    X_val = torch.randn(bsz, qlen, hd, dtype=torch.bfloat16, device=device, requires_grad=True)
    W_val = torch.randn(vocab, hd, dtype=torch.bfloat16, device=device, requires_grad=True)
    if label_dim == 2:
        labels_val = torch.randint(0, vocab, (bsz * qlen,), device=device)
    elif label_dim == 3:
        labels_val = torch.randn(bsz, qlen, vocab, dtype=torch.bfloat16, device=device)
    else:
        raise ValueError("label_dim must be 2 or 3.")

    torch.cuda.reset_peak_memory_stats(device)
    loss_ref = loss_fn(X_val, W_val, labels_val)
    loss_ref.backward()
    ref_peak = torch.cuda.max_memory_allocated(device)
    ref_dX = X_val.grad.clone()
    ref_dW = W_val.grad.clone()

    X_val.grad.zero_()
    W_val.grad.zero_()

    torch.cuda.reset_peak_memory_stats(device)
    loss_chunked = MemoryEfficientLinear.apply(X_val, W_val, labels_val, transformation_fn, 1, 256)
    loss_chunked.backward()
    chunk_peak = torch.cuda.max_memory_allocated(device)
    reduction = (ref_peak - chunk_peak) / ref_peak * 100 if ref_peak != 0 else 0

    print(f"{loss_name} Reference Loss: {loss_ref.item():.4f}")
    print(f"{loss_name} Chunked Loss:   {loss_chunked.item():.4f}")
    print("X gradients match? ", torch.allclose(X_val.grad, ref_dX, atol=1e-3))
    print("W gradients match? ", torch.allclose(W_val.grad, ref_dW, atol=1e-3))
    print("Reference peak memory (bytes):", ref_peak)
    print("Chunked peak memory (bytes):  ", chunk_peak)
    print(f"Percent VRAM reduction: {reduction:.2f}%\n")

### Main Testing ###

if __name__ == "__main__":
    # Full test for CE.
    bsz, qlen, hd, vocab = 4, 4096, 4096, 128000
    X = torch.randn(bsz, qlen, hd, dtype=torch.bfloat16, device=device, requires_grad=True)
    W = torch.randn(vocab, hd, dtype=torch.bfloat16, device=device, requires_grad=True)
    labels_CE = torch.randint(0, vocab, (bsz * qlen,), device=device)
    loss_CE = MemoryEfficientLinear.apply(X, W, labels_CE, transformation_function_CE, 1, 1024)
    loss_CE.backward()
    print("CE Test Loss:", loss_CE.item())
    print("Gradients for X computed:", X.grad is not None)
    print("Gradients for W computed:", W.grad is not None)
    print()

    print("Validating Cross Entropy (CE) version:")
    validate_GRPO_memory_efficient_linear(reference_loss_fn_CE, transformation_function_CE, label_dim=2, loss_name="CE")

    print("Validating Focal (other function) version:")
    validate_GRPO_memory_efficient_linear(lambda X,W,labels: reference_loss_fn_focal(X,W,labels, gamma=1.0, eps=1e-7),
                                            lambda b,w,l: transformation_function_focal(b,w,l, gamma=1.0, eps=1e-7),
                                            label_dim=2,
                                            loss_name="Focal")
    
    print("Validating Llama-1B Training Loss Matching:")
    validate_llama_training_loss_matches()

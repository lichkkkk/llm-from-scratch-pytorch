import torch

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    '''
    'batch vocab, batch' -> ''
    '''
    max_logits = inputs.max(dim=1, keepdim=True).values
    rebased_inputs = inputs - max_logits
    target_logits = rebased_inputs[torch.arange(inputs.shape[0]), targets]
    logsumexp = torch.log(torch.sum(torch.exp(rebased_inputs), dim=1))
    return (logsumexp - target_logits).mean()

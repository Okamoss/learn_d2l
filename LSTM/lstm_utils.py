import torch


def grad_clipping(net, theta):
    """Clip gradients to avoid exploding gradients in RNNs."""
    params = [p for p in net.parameters() if p.requires_grad]
    if not params:
        return
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
    if norm.item() > theta:
        scale = theta / norm.item()
        for param in params:
            param.grad.mul_(scale)


def predict_rnn(prefix, num_preds, net, vocab, device):
    """Generate text with the trained model."""
    net.eval()
    state = net.begin_state(batch_size=1, device=device)

    output_indices = [vocab[prefix[0]]]

    for i in range(len(prefix) - 1):
        input_tensor = torch.tensor([vocab[prefix[i]]], device=device).reshape(1, 1)
        _, state = net(input_tensor, state)

    input_tensor = torch.tensor([vocab[prefix[-1]]], device=device).reshape(1, 1)
    for _ in range(num_preds):
        output, state = net(input_tensor, state)
        pred_index = output.argmax(dim=1).item()
        output_indices.append(pred_index)
        input_tensor = torch.tensor([pred_index], device=device).reshape(1, 1)

    if hasattr(vocab, "to_tokens"):
        return "".join(vocab.to_tokens(output_indices))
    if hasattr(vocab, "idx_to_token"):
        return "".join([vocab.idx_to_token[i] for i in output_indices])
    return str(output_indices)
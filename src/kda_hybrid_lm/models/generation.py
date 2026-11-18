import torch


@torch.inference_mode()
def generate(model, input_ids, max_new_tokens: int):
    """固定长度贪心生成，使用各层缓存并恢复调用前的训练模式。"""
    was_training = model.training
    model.eval()
    tokens = input_ids
    state = None
    current = input_ids
    try:
        for _ in range(max_new_tokens):
            output = model(current, state=state, use_cache=True)
            last_logits = output.logits[:, -1]
            current = last_logits.argmax(dim=-1, keepdim=True)
            tokens = torch.cat((tokens, current), dim=1)
            state = output.state
    finally:
        model.train(was_training)
    return tokens

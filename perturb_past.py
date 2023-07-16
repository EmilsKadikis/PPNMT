from typing import List, Tuple
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import PreTrainedModel
import numpy as np
from operator import add

SMALL_CONST = 1e-15
BIG_CONST = 1e10

class PerturbationArgs:
    def __init__(self, **kwargs):
        # Parameters that control the length of the output
        self.decay = kwargs.pop("decay", True)
        self.window_length = kwargs.pop("window_length", 5)
        self.num_iterations = kwargs.pop("num_iterations", 6)
        self.positive_bag_of_words = kwargs.pop("positive_bag_of_words", None)
        self.negative_bag_of_words = kwargs.pop("negative_bag_of_words", None)
        self.gamma = kwargs.pop("gamma", 1)
        self.kl_scale = kwargs.pop("kl_scale", 0.1)
        self.gm_scale = kwargs.pop("gm_scale", 0.95)
        self.stepsize = kwargs.pop("stepsize", 0.1)
        self.temperature = kwargs.pop("temperature", 1)
        self.grad_length = kwargs.pop("grad_length", 8)

def _to_var(x, requires_grad=False, volatile=False, device='cuda'):
    if torch.cuda.is_available() and device == 'cuda':
        x = x.cuda()
    elif device != 'cuda':
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

def _get_decay_mask(decay: bool, window_length: int):
    if decay and window_length > 0:
        decay_mask = torch.arange(
            0.,
            1.0 + SMALL_CONST,
            1.0 / (window_length)
        )[1:] # [0.2, 0.4, 0.6, 0.8, 1] shape [window_length]
    else:
        decay_mask = 1.0
    return decay_mask

def _get_window_mask(curr_length, window_length, decay, attention, device='cuda'):
    decay_mask = _get_decay_mask(decay, window_length)
    if curr_length > window_length and window_length > 0:
        ones_key_val_shape = (
                tuple(attention[0].shape[:-2])
                + tuple([window_length])
                + tuple(attention[0].shape[-1:])
        )

        zeros_key_val_shape = (
                tuple(attention[0].shape[:-2])
                + tuple([curr_length - window_length])
                + tuple(attention[0].shape[-1:])
        )

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

        window_mask = torch.cat(
            (ones_mask, torch.zeros(zeros_key_val_shape)),
            dim=-2
        ).to(device)
    else:
        window_mask = torch.ones_like(attention[0]).to(device) # [2, batch, num_heads, seq_len, head_dim]

    return window_mask

def _add_bag_of_word_loss(losses, probs: torch.Tensor, args: PerturbationArgs):
    # extend one_hot_bow to the batch size
    if args.positive_bag_of_words is not None:
        bow_logits = torch.mm(probs, torch.t(args.positive_bag_of_words))
        bow_loss = -torch.log(torch.sum(bow_logits, dim=1))
        losses += bow_loss
    if args.negative_bag_of_words is not None:
        bow_logits = torch.mm(probs, torch.t(args.negative_bag_of_words))
        bow_loss = torch.log(torch.sum(bow_logits, dim=1)) # The loss here has the oppsoite sign
        losses += bow_loss

def _add_kl_loss(loss, probs: torch.Tensor, unperturbed_logits: torch.Tensor, args: PerturbationArgs):
    # probs shape: [batch, vocab_size]
    # unperturbed_logits shape: [batch, seq_len, vocab_size]
    device = probs.device
    if args.kl_scale > 0.0:
        unpert_probs = F.softmax(unperturbed_logits[:, -1, :], dim=-1) # [batch, vocab_size], gets the probabilities of the last token
        unpert_probs = (
                unpert_probs + SMALL_CONST *
                (unpert_probs <= SMALL_CONST).float().to(device).detach()
        ) # adds a small constant to the probabilities that are less than a small constant
        correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(
            device).detach() # [batch, vocab_size]. All zeroes except for the spots that were too small, those are equal to SMALL_CONST
        # Both things above are done to avoid log(0)
        corrected_probs = probs + correction.detach()
        kl_loss = args.kl_scale * (
            (corrected_probs * (corrected_probs / unpert_probs).log()).sum()
        )
        loss += kl_loss

def perturb_past(
    past: Tuple[Tuple[torch.Tensor]], # past[6][4] (six layers, key and values of self attn and cross attn), shape [batch, num_heads, seq_len, head_dim]
    last_tokens: torch.Tensor, # shape [batch, 1]
    encoder_hidden_states: torch.Tensor, # shape [batch, input_len, hidden_size]
    model: PreTrainedModel,
    unperturbed_logits: torch.Tensor, # shape [batch, seq_len, vocab_size]
    grad_norms_self_attn: List[torch.Tensor],
    encoder_attention_mask: torch.Tensor, # shape [batch, seq_len]
    args: PerturbationArgs,
) -> Tuple[Tuple[torch.Tensor]]:  
    if past is None or last_tokens is None:
        return past, grad_norms_self_attn
    
    # do perturbation
    past_self_attn = [torch.cat((p[0].unsqueeze(0), p[1].unsqueeze(0)), dim=0) for p in past]
    past_cross_attn = [torch.cat((p[2].unsqueeze(0), p[3].unsqueeze(0)), dim=0) for p in past]
    device = past_self_attn[0].device
    
    _, _, _, curr_length, _ = past_self_attn[0].shape

    if curr_length >= args.grad_length:
        return past, grad_norms_self_attn

    grad_accumulator_self_attn = [
        (np.zeros(p.shape).astype("float32"))
        for p in past_self_attn
    ] # [2, batch_size, num_heads, current_seq_len, head_dim] (the 2 is for key and value of self attention)

    window_mask = _get_window_mask(curr_length, args.window_length, args.decay, past_self_attn, device)

    for i in range(args.num_iterations):
        curr_perturbation_self_attn = [
            _to_var(torch.from_numpy(p_), requires_grad=True, device=device)
            for p_ in grad_accumulator_self_attn  # p_ one for each layer of the decoder
        ] # converted to a tensor so that we can do gradient descent on it?
        
        perturbed_past_self_attn = list(map(add, past_self_attn, curr_perturbation_self_attn)) # for each layer, add the perturbation to the past
        
        # reconstruct the past in the format that the model expects
        perturbed_past = [(p_self[0], p_self[1], p_cross[0], p_cross[1]) for p_self, p_cross in zip(perturbed_past_self_attn, past_cross_attn)]
        
        if last_tokens is None:
            last_tokens = torch.LongTensor([[model.config.pad_token_id]] * past_self_attn[0].shape[1]).to(device)
        elif len(last_tokens.shape) == 1:
            last_tokens = last_tokens.unsqueeze(-1) # [batch, 1]
        model_output = model.get_decoder()(input_ids=last_tokens, past_key_values=perturbed_past, encoder_hidden_states=encoder_hidden_states[-1], encoder_attention_mask=encoder_attention_mask)
        lm_logits = model.lm_head(model_output[0]) + model.final_logits_bias
        logits = lm_logits[:, -1, :] # [batch, vocab_size] (takes the logits of the last token)
        probs = F.softmax(logits, dim=-1) # [batch, vocab_size]

        batch_size = probs.shape[0]
        loss = torch.zeros(batch_size).to(device)
        _add_bag_of_word_loss(loss, probs, args)
        _add_kl_loss(loss, probs, unperturbed_logits, args)

        loss.sum().backward(retain_graph=True)

        # Compute gradient norms
        if grad_norms_self_attn is None:
            grad_norms_self_attn = [torch.zeros(batch_size, device=device) for _ in curr_perturbation_self_attn]
        grad_norms_self_attn = [
            torch.max(grad_norms_self_attn[index], torch.norm(p_.grad * window_mask))
            for index, p_ in enumerate(curr_perturbation_self_attn)
        ]

        # Calculate final gradients
        grad_self_attn = [
            -args.stepsize *
            ((p_.grad * window_mask) / grad_norms_self_attn[ #[2, 3, 8, 1, 64] / [3]
                index].unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) ** args.gamma).data.cpu().numpy()
            for index, p_ in enumerate(curr_perturbation_self_attn)
        ]

        # accumulate gradient
        grad_accumulator_self_attn = list(map(add, grad_self_attn, grad_accumulator_self_attn))

    grad_accumulator_self_attn = [
        _to_var(torch.from_numpy(p_), requires_grad=True, device=device)
        for p_ in grad_accumulator_self_attn
    ]
    perturbed_past_self_attn = list(map(add, past_self_attn, grad_accumulator_self_attn))
    # perturbed_past_cross_attn = list(map(add, past_cross_attn, grad_accumulator_cross_attn))
    perturbed_past = [[p_self[0], p_self[1], p_cross[0], p_cross[1]] for p_self, p_cross in zip(perturbed_past_self_attn, past_cross_attn)]

    return perturbed_past, grad_norms_self_attn
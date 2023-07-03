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

def perturb_past(
    past: Tuple[Tuple[torch.Tensor]], # past[6][4] (six layers, key and values of self attn and cross attn), shape [batch, num_heads, seq_len, head_dim]
    last_token: torch.Tensor, # shape [batch, 1]
    encoder_hidden_states: torch.Tensor, # shape [batch, input_len, hidden_size]
    model: PreTrainedModel,
    args: PerturbationArgs,
) -> Tuple[Tuple[torch.Tensor]]:  
    # do perturbation
    past_self_attn = [torch.cat((p[0].unsqueeze(0), p[1].unsqueeze(0)), dim=0) for p in past]
    past_cross_attn = [torch.cat((p[2].unsqueeze(0), p[3].unsqueeze(0)), dim=0) for p in past]
    device = past_self_attn[0].device
    
    _, _, _, curr_length, _ = past_self_attn[0].shape

    grad_accumulator_self_attn = [
        (np.zeros(p.shape).astype("float32"))
        for p in past_self_attn
    ] # [2, batch_size, num_heads, current_seq_len, head_dim] (the 2 is for key and value of self attention)

    decay_mask = _get_decay_mask(args.decay, args.window_length)
    window_mask_ = _get_window_mask(curr_length, args.window_length, past_self_attn, decay_mask, device)

    for i in range(args.num_iterations):
        curr_perturbation_self_attn = [
            _to_var(torch.from_numpy(p_), requires_grad=True, device=device)
            for p_ in grad_accumulator_self_attn  # p_ one for each layer of the decoder
        ] # converted to a tensor so that we can do gradient descent on it?
        
        perturbed_past_self_attn = list(map(add, past_self_attn, curr_perturbation_self_attn)) # for each layer, add the perturbation to the past
        
        # reconstruct the past in the format that the model expects
        perturbed_past = [(p_self[0], p_self[1], p_cross[0], p_cross[1]) for p_self, p_cross in zip(perturbed_past_self_attn, past_cross_attn)]
        
        model_output = model.get_decoder()(input_ids=last_token, past_key_values=perturbed_past, encoder_hidden_states=encoder_hidden_states)
        lm_logits = model.lm_head(model_output[0]) + model.final_logits_bias
        logits = lm_logits[:, -1, :] # [batch, vocab_size] (takes the logits of the last token)
        probs = F.softmax(logits, dim=-1) # [batch, vocab_size]

        loss = 0.0
        _add_bag_of_word_loss(loss, probs, args)
        _add_kl_loss(loss, probs, args)

    perturbed_past = past
    return perturbed_past

def _add_bag_of_word_loss(loss, probs: torch.Tensor, args: PerturbationArgs):
    for one_hot_bow in args.positive_bag_of_words:
        bow_logits = torch.mm(probs, torch.t(one_hot_bow))
        bow_loss = -torch.log(torch.sum(bow_logits))
        loss += bow_loss
    for one_hot_bow in args.negative_bag_of_words:
        bow_logits = torch.mm(probs, torch.t(one_hot_bow))
        bow_loss = torch.log(torch.sum(bow_logits))  # The loss here has the oppsoite sign
        loss += bow_loss

def _add_kl_loss(loss, probs: torch.Tensor, args: PerturbationArgs):
    device = probs.device
    if args.kl_scale > 0.0:
        unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1) # [batch, vocab_size], gets the probabilities of the last token
        unpert_probs = (
                unpert_probs + SMALL_CONST *
                (unpert_probs <= SMALL_CONST).float().to(device).detach()
        ) # adds a small constant to the probabilities that are less than a small constant
        correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(
            device).detach() # [batch, vocab_size]. All zeroes except for the spots that were too small, those are equal to SMALL_CONST
        # The two statements above seem to do the same exact thing.
        # They're done to avoid log(0)
        corrected_probs = probs + correction.detach()
        kl_loss = kl_scale * (
            (corrected_probs * (corrected_probs / unpert_probs).log()).sum()
        )
        if verbosity_level >= VERY_VERBOSE:
            print(' kl_loss', kl_loss.data.cpu().numpy() / kl_scale)
        loss += kl_loss
        loss_list.append(kl_loss)


def _get_decay_mask(decay: bool, window_length: int):
    if decay:
        decay_mask = torch.arange(
            0.,
            1.0 + SMALL_CONST,
            1.0 / (window_length)
        )[1:] # [0.2, 0.4, 0.6, 0.8, 1] shape [window_length]
    else:
        decay_mask = 1.0
    return decay_mask

def _get_window_mask(curr_length, window_length, attention, decay_mask, device='cuda'):
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


def _to_var(x, requires_grad=False, volatile=False, device='cuda'):
    if torch.cuda.is_available() and device == 'cuda':
        x = x.cuda()
    elif device != 'cuda':
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


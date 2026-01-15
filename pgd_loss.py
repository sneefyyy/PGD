import abc
import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification

from experiment import experiment
from .pgd_utils import append_or_allocate, tsallis, TOKEN_LVL_, STEP_LVL_
from .prompt import PromptManager


def pos_mean_p_norm(values: torch.Tensor,
                    p: int,
                    weights: torch.Tensor | None = None,
                    all_values_positive: bool = True,
                    dim: int = -1) -> torch.Tensor:
    """Calculates the p norm and normalizes ist by 1 / N^(1/p). For `p=1` it
    assumes that all values are positive.

    Args:
        values (torch.Tensor): Values to be normed, by default choice of `dim`
            of shape [..., N].
        p (int): Norm to apply.
        weights (torch.Tensor, optional): How to weight each term, by default
            choice of `dim` of shape [..., N].
        all_values_positive (bool, optional): If True all values are
            assumed to be positive.
        dim (int, optional): Default dimension to apply norm. Defaults to -1.

    Returns:
        torch.Tensor: normed and averaged value.
    """
    if p == 1 and all_values_positive:
        if weights is None:
            return values.mean(dim)
        else:
            return (values * weights).sum(dim)

    if weights is None:
        norm = torch.linalg.vector_norm(values, ord=p, dim=dim)
        mean_norm = norm / (values.shape[dim]**(1 / p))
        return mean_norm

    eps = torch.finfo(values.dtype).smallest_normal
    values_pp = values**p
    values_pp = (values_pp * weights).sum(dim)
    mean_norm = torch.clamp_min(values_pp, eps)**(1 / p)
    return mean_norm


class PGDLoss(nn.Module, abc.ABC):

    @experiment.capture(prefix='attack')
    def __init__(self,
                 target_weight: float = 0.84,
                 aux_target_weight: float = 0.,
                 control_weight: float = 0.007,
                 control_next_weight: float = 0.05,
                 control_nonrepeat_weight: float = 0.01,
                 entropy_weight: float = 2e-4,
                 length_weight: float = 0.,
                 judge_weight: float = 1.5,
                 loss_temperature=1.,
                 target_p=1.,
                 control_p=1.,
                 entropy_q=2.,
                 entropy_p=6.,
                 token_position_weighting_kwargs={
                     'name': 'linear',  # linear or uniform
                     'first_last_ratio': 5.
                 },
                 verbose: int = 1,
                 **kwargs) -> None:
        if kwargs:
            print('Loss kwargs are unused', kwargs)

        super().__init__()
        self.target_weight = target_weight
        self.aux_target_weight = aux_target_weight
        self.control_weight = control_weight
        self.control_next_weight = control_next_weight
        self.control_nonrepeat_weight = control_nonrepeat_weight
        self.entropy_weight = entropy_weight
        self.length_weight = length_weight
        self.judge_weight = judge_weight
        self.loss_temperature = loss_temperature
        self.target_p = target_p
        self.control_p = control_p
        self.entropy_q = entropy_q
        self.entropy_p = entropy_p
        self.token_position_weighting_kwargs = token_position_weighting_kwargs
        self.verbose = verbose

        self.ce = nn.CrossEntropyLoss(reduction='none')

    @abc.abstractmethod
    def print(self, *args, level=logging.INFO):
        ...

    def control_loss(self,
                     factors: torch.Tensor | None,
                     mask: torch.Tensor | None,
                     logits: torch.Tensor,
                     ids: torch.Tensor | None,
                     loss_dict_prefix: str,
                     log: dict | None = None,
                     logver_prefix: str = 'relaxed'):
        weights = mask
        if weights is not None:
            weights = weights / weights.sum(-1, keepdims=True)
        else:
            weights = torch.full_like(logits[..., 0, :], 1. / logits.shape[-1])

        control_loss = 0
        if self.control_weight != 0 and factors is not None:
            control_loss_ = self.ce(logits, factors.transpose(1, 2).detach())
            control_loss = pos_mean_p_norm(control_loss_, self.control_p, weights=weights)

        control_next_loss = 0
        if self.control_next_weight != 0 and factors is not None:
            control_loss_ = self.ce(logits.detach(), factors.transpose(1, 2))
            control_next_loss = pos_mean_p_norm(control_loss_, self.control_p, weights=weights)

        if ((self.control_weight != 0 or self.control_next_weight != 0) and
                factors is None):
            control_loss_ = self.ce(logits, ids)
            control_loss = pos_mean_p_norm(control_loss_, self.control_p, weights=weights)

        control_nonrepeat_loss = 0
        if self.control_nonrepeat_weight != 0 and factors is not None:
            # This is not perfectly accurate if optimizing over variable length
            control_nonrepeat_loss = -torch.linalg.norm(
                factors[:, :-1, :].detach() - factors[:, 1:, :], dim=-1, ord=1).mean(-1)

        loss_joint, loss_dict_joint = self.joint_loss(
            factors, mask, logits, ids, loss_dict_prefix, log, logver_prefix)

        loss = (self.control_weight * control_loss +
                self.control_next_weight * control_next_loss +
                self.control_nonrepeat_weight * control_nonrepeat_loss +
                loss_joint)

        perplexity = 0
        if self.control_weight != 0 or self.control_next_weight != 0:
            with torch.no_grad():
                perplexity = (weights * control_loss_).sum(-1).exp()
                # perplexity = detach_to_cpu(perplexity)

        # control_loss = detach_to_cpu(control_loss)
        loss_dict = {f'{loss_dict_prefix}': control_loss,
                     f'{loss_dict_prefix}_perplexity': perplexity}
        loss_dict.update(loss_dict_joint)

        if log is not None and self.verbose > 1:
            if self.control_weight != 0 or self.control_next_weight != 0:
                append_or_allocate(log[TOKEN_LVL_], f'loss_{logver_prefix}_loss', control_loss_)

        return loss, loss_dict

    def target_loss(self,
                    factors: torch.Tensor | None,
                    mask: torch.Tensor | None,
                    logits: torch.Tensor,
                    target_ids: torch.Tensor,
                    prompt_mgr: PromptManager,
                    loss_dict_prefix: str,
                    log: dict | None = None,
                    logver_prefix: str = 'relaxed'):
        if factors is not None:
            # Remain close to what is likely for the model if optimizing target
            target_loss_ = self.ce(logits, factors.transpose(1, 2))
            target_weights = self.get_target_weights(mask)
        else:
            # Loss towards the affirmative response
            target_ids = target_ids.long().to(logits.device)
            target_loss_ = self.ce(logits, target_ids)
            target_weights = self.get_target_weights(mask)

        target_loss = pos_mean_p_norm(target_loss_, self.target_p, weights=target_weights)

        loss = self.target_weight * target_loss
        loss_dict = {f'{loss_dict_prefix}': target_loss}

        if factors is not None:
            loss_joint, loss_dict_joint = self.joint_loss(
                factors, mask, logits, target_ids, loss_dict_prefix, log, logver_prefix)
            loss += loss_joint
            loss_dict.update(loss_dict_joint)

        if log is not None and self.verbose > 1:
            append_or_allocate(log[TOKEN_LVL_], f'loss_{logver_prefix}_target', target_loss_)

        return loss, loss_dict

    def joint_loss(self,
                   factors: torch.Tensor | None,
                   mask: torch.Tensor | None,
                   logits: torch.Tensor,
                   ids: torch.Tensor | None,
                   loss_dict_prefix: str,
                   log: dict | None = None,
                   logver_prefix: str = 'relaxed'):
        weights = mask
        if weights is not None:
            weights = weights / weights.sum(-1, keepdims=True)
        else:
            weights = torch.full_like(logits[..., 0, :], 1. / logits.shape[-1])

        entropy_loss = 0
        if self.entropy_weight != 0 and factors is not None:
            # Detach to avoid suppressing low nn/entropy tokens
            mask_weights_d = weights.detach()
            entropy_loss_ = tsallis(factors, q=self.entropy_q)
            entropy_loss = pos_mean_p_norm(entropy_loss_, self.entropy_p, weights=mask_weights_d)

        length_loss = 0
        if mask is not None:
            length_loss = pos_mean_p_norm(mask.float(), 1)

        loss = (self.entropy_weight * entropy_loss +
                self.length_weight * length_loss)
        loss_dict = {
            f'{loss_dict_prefix}_entropy': entropy_loss,
            f'{loss_dict_prefix}_length': length_loss,
        }

        if log is not None and self.verbose > 1:
            if self.entropy_weight != 0 and factors is not None:
                append_or_allocate(log[TOKEN_LVL_], f'loss_{logver_prefix}_entropy', entropy_loss_)

        return loss, loss_dict

    def loss(self,
             logits,
             prompt_mgr: PromptManager,
             control_and_target: Tuple[torch.Tensor | None, ...],
             ids: torch.Tensor | None = None,
             attention_mask: torch.Tensor | None = None,
             log: dict | None = None,
             logver_prefix: str = 'relaxed') -> Dict[str, torch.Tensor]:

        logits_raw = logits
        if self.loss_temperature != 1.:
            logits = logits / self.loss_temperature

        # Target & judge loss
        factors, mask = control_and_target[-2:]
        target_logits = logits[prompt_mgr.target_batch_id, prompt_mgr.target_tok_id - 1]
        target_logits = target_logits.transpose(1, 2)
        if factors is not None:
            target_ids = factors.argmax(-1)
        elif ids is not None:
            target_ids = ids[prompt_mgr.target_batch_id, prompt_mgr.target_tok_id]
        else:
            target_ids = prompt_mgr.input_ids.to(logits.device)
            target_ids = target_ids[prompt_mgr.target_batch_id, prompt_mgr.target_tok_id]

        target_id_mask = prompt_mgr.get_target_mask().to(logits.device)
        if mask is None:
            mask = target_id_mask
        target_weights = mask / mask.sum(-1, keepdims=True)  # required below

        loss, loss_dict = self.target_loss(
            factors, mask, target_logits, target_ids, prompt_mgr, 'target', log, logver_prefix + '_target')

        # Prefix loss
        cp_loss, cp_loss_dict = 0, {}
        if prompt_mgr.control_prefix_length:
            factors, mask = control_and_target[:2]
            control_logits = logits[prompt_mgr.c_prefix_batch_id, prompt_mgr.c_prefix_tok_id - 1]
            control_logits = control_logits.transpose(1, 2)
            if ids is None:
                control_ids = None
            else:
                control_ids = ids[prompt_mgr.c_prefix_batch_id, prompt_mgr.c_prefix_tok_id]

            cp_loss, cp_loss_dict = self.control_loss(
                factors, mask, control_logits, control_ids, 'control_prefix',
                log, logver_prefix + '_control_prefix')

        # Suffix loss
        cs_loss, cs_loss_dict = 0, {}
        if prompt_mgr.control_suffix_length:
            factors, mask = control_and_target[2:4]
            control_logits = logits[prompt_mgr.c_suffix_batch_id, prompt_mgr.c_suffix_tok_id - 1]
            control_logits = control_logits.transpose(1, 2)
            if ids is None:
                control_ids = None
            else:
                control_ids = ids[prompt_mgr.c_suffix_batch_id, prompt_mgr.c_suffix_tok_id]

            cs_loss, cs_loss_dict = self.control_loss(
                factors, mask, control_logits, control_ids, 'control_suffix', log, logver_prefix + '_control_suffix')

        # Combine loss terms
        loss += cp_loss + cs_loss

        loss_dict['combined'] = loss
        loss_dict.update(cp_loss_dict)
        loss_dict.update(cs_loss_dict)

        # Log loss-like metrics
        with torch.no_grad():
            target_logits_raw = logits_raw[prompt_mgr.target_batch_id, prompt_mgr.target_tok_id - 1]
            ce_loss_raw = self.ce(target_logits_raw.transpose(1, 2), target_ids)
            target_ce = (ce_loss_raw * target_weights).sum(-1)
            em_prob = torch.exp(-target_ce * (target_weights > 0).sum(-1))
            perplexity = (target_weights * ce_loss_raw).sum(-1).exp()

            loss_dict['target_ce'] = target_ce
            loss_dict['target_em_prob'] = em_prob
            # backwards compatibility
            loss_dict['target_em_wkt'] = em_prob
            loss_dict['target_perplexity'] = perplexity

        if log is not None:
            for key, value in loss_dict.items():
                if not isinstance(value, torch.Tensor):
                    continue
                append_or_allocate(log[STEP_LVL_], f'loss_{logver_prefix}_{key}', value)

        if self.verbose > 1:
            # self.print(f'{logver_prefix} token lvl target_loss',
            #            [[f'{v:.3g}' for v in row]
            #             for row in target_loss_.detach().cpu().tolist()])
            self.print(
                logver_prefix,
                *(a for k, v_arr in loss_dict.items() if isinstance(v_arr, torch.Tensor) for a in (k, v_arr.tolist())))

        return loss_dict

    @torch.no_grad()
    def get_target_weights(self, target_mask):
        device = target_mask.device
        n_tokens = target_mask.sum(-1, keepdims=True)

        func_name = self.token_position_weighting_kwargs.get('name', 'uniform')

        if func_name == 'uniform':
            return target_mask / n_tokens
        elif func_name == 'linear':
            intercept = self.token_position_weighting_kwargs['first_last_ratio']
            slope = (1 - intercept) / (n_tokens - 1)
            target_weights = torch.arange(target_mask.shape[-1], device=device)
            target_weights = intercept + slope * target_weights[None]
            target_weights = target_weights * target_mask
            target_weights /= target_weights.sum(-1, keepdims=True)
            return target_weights
        else:
            raise ValueError(f"Unknown token weighting function {func_name}")
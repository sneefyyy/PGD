import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F
import wandb

from experiment import experiment
from ..baseline import RedTeamingMethod
from ..check_refusal_utils import check_refusal_completions
from .pgd_loss import PGDLoss
from .prompt import PromptManager
from .pgd_utils import (
    tsallis, x_bounded_sigmoid, EarlyStoppingAndPatience, LogarithmicLR, maybe_broadcast_token_lvl_logs,
    aggregate_logs, generate_wandb_plots, get_nonascii_toks, get_embedding_matrix, one_hot_tensor,
    FINAL_, TOKEN_LVL_, STEP_LVL_, ATTACK_LVL_, append_or_allocate)

torch.set_float32_matmul_precision('medium')

# Configure the logging format
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    level=logging.INFO
)


class PGDMultiPromptAttack(PGDLoss, RedTeamingMethod):

    @experiment.capture(prefix='attack')
    def __init__(
            self,
            model,
            template,
            tokenizer,
            strategy='pgd',  # Options `gbda` or `pgd`
            num_steps=5_000,
            eval_best=False,
            eval_steps=250,
            learning_rate=0.11,
            grad_clip_value=20,
            grad_clip_strategy='token_norm',
            entropy_factor: float | torch.Tensor = 0.4,
            allow_non_ascii=False,
            anneal=True,
            # only relevant if strategy is `gbda`
            gbda_config=dict(temp=1., deterministic=True, soft_frac=0.72,
                             anneal_init_temp=100, anneal_end_temp=0.001, gumble_hard=False),
            anneal_config=dict(
                start=0,
                duration=250,
                attrs=['entropy_factor'],
                mode='uniform',  # 'uniform', 'lin_order'
                init_entropy_factor=0.,
                end_entropy_factor=0.4
            ),
            simplex_proj_method='sort',  # bisection or sort
            tsallis_q2_proj_config=dict(iter=1, exclude_already_zero=True),
            optimizer_config=dict(
                name='Adam', param_groups={'embedding_factors': {}, 'embedding_mask': {'lr': 0.02}},),
            lr_scheduler_config=dict(
                name='SequentialLR', milestones=[100], schedulers=[
                    dict(name='ConstantLR', factor=1),
                    dict(name='CosineAnnealingWarmRestarts', T_0=60, eta_min=0.325)]),
            variable_control_length=dict(enable=False, scale_entropy_factor='lin', max_anneal_temperature=1.075),
            entropy_factor_scale_by_relaxation_gap=0.1,
            entropy_factor_alternate_scheduler=True,
            patience_config=dict(value=100, patience_mix_probability=0.5, patience_mix_temp=0.25),
            early_stop_key='target_ce',
            early_stop_minimize=True,
            control_prefix_length: int | None = 25,
            control_suffix_length: int = 25,
            control_prefix_init: List[str] | None = None,
            control_suffix_init: List[str] | None = None,
            target_length: int | None = None,
            langevin_dynamics_std: float | None = None,
            initialization: str = 'random',
            use_prefix_cache: bool = True,  # only for hugging face models
            **kwargs):
        """Attack a model using the Projected Gradient Descent (PGD) method using the affirmative objective.

        :param model: LLM to attack
        :param template: model-specific prompt template
        :param tokenizer: tokenizer for LLM to attack
        :param strategy: either 'pgd' or 'gbda', defaults to 'pgd'
        :param eval_best: if true, always evaluate best (according to early_stopping_key), defaults to False
        :param eval_steps: how often to run eval, defaults to 250
        :param learning_rate: base learning rate, defaults to 0.11
        :param grad_clip_value: gradient clipping magnitide, defaults to 20
        :param grad_clip_strategy: either global 'norm', token level norm 'token_norm' or [-value, value] range 'value', defaults to 'token_norm'
        :param entropy_factor: strength of entropy projection, defaults to 0.4
        :param allow_non_ascii: if true also tokens with non-ascii chars, defaults to False
        :param anneal: if true activate anneling, defaults to True
        :param gbda_config: GBDA specific configurations, defaults to dict(temp=1., deterministic=True, soft_frac=0.72, anneal_init_temp=100, anneal_end_temp=0.001, gumble_hard=False)
        :param anneal_config: how to anneal arbitrary class params, defaults to dict( start=0, duration=250, attrs=['entropy_factor'], mode='uniform', init_entropy_factor=0., end_entropy_factor=0.4 )
        :param simplex_proj_method: use either 'bisection'- or 'sort'ing-based projection, defaults to 'sort'
        :param optimizer_config: config for PyTorch optimizer, defaults to dict( name='Adam', param_groups={'embedding_factors': {}, 'embedding_mask': {'lr': 0.02}},)
        :param lr_scheduler_config: config for PyTorch learning rate scheduler, defaults to dict( name='SequentialLR', milestones=[100], schedulers=[ dict(name='ConstantLR', factor=1), dict(name='CosineAnnealingWarmRestarts', T_0=60, eta_min=0.325)])
        :param variable_control_length: enable optimizing variable length prompts (not implemented for HuggingFace models), defaults to dict(enable=False, scale_entropy_factor='lin', max_anneal_temperature=1.075)
        :param entropy_factor_scale_by_relaxation_gap: weaken entropy projection if relaxed and discrete loss are close, defaults to 0.1
        :param entropy_factor_alternate_scheduler: couple entropy projection to learning rate scheduler, defaults to True
        :param patience_config: reset to last best if out of patience and mix adversarial suffixes between prompts, defaults to dict(value=100, patience_mix_probability=0.5, patience_mix_temp=0.25)
        :param early_stop_key: metric used for early stopping, defaults to 'target_ce'
        :param early_stop_minimize: lower is better if True, defaults to True
        :param control_prefix_length: length of adversarial prefix (<= 0 to deactivate), defaults to 25
        :param control_suffix_length: length of adversarial suffix, defaults to 25
        :param control_prefix_init: list of strings for init, defaults to None
        :param control_suffix_init: list of strings for init, defaults to None
        :param target_length: limit on target length, defaults to None
        :param langevin_dynamics_std: optionally also apply randomness to gradient, defaults to None
        :param initialization: either init randomly or with discrete tokens, defaults to 'random'
        :param use_prefix_cache: whether to use KV cache for static prefix tokens (WARNING: Not all models support this), defaults to True
        """
        # For passing arguments to the loss function
        if kwargs:
            print('PGD loss kwargs', kwargs)

        PGDLoss.__init__(self, **kwargs)

        self.template = template
        self.tokenizer = tokenizer
        self.num_steps = num_steps
        self.strategy = strategy
        self.gbda_config = gbda_config
        self.eval_best = eval_best
        self.eval_steps = eval_steps
        self.learning_rate = learning_rate
        self.grad_clip_value = grad_clip_value
        self.grad_clip_strategy = grad_clip_strategy
        self.allow_non_ascii = allow_non_ascii
        self.anneal = anneal
        self.entropy_factor = entropy_factor
        self.anneal_config = anneal_config
        self.simplex_proj_method = simplex_proj_method
        self.tsallis_q2_proj_config = tsallis_q2_proj_config
        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config
        self.variable_control_length = variable_control_length
        self.entropy_factor_scale_by_relaxation_gap = entropy_factor_scale_by_relaxation_gap
        self.entropy_factor_alternate_scheduler = entropy_factor_alternate_scheduler
        self.patience_config = patience_config
        self.control_prefix_length = control_prefix_length
        self.control_suffix_length = control_suffix_length
        self.control_prefix_init = control_prefix_init
        self.control_suffix_init = control_suffix_init
        self.target_length = target_length
        self.initialization = initialization
        self.langevin_dynamics_std = langevin_dynamics_std
        self.use_prefix_cache = use_prefix_cache

        self.esp = EarlyStoppingAndPatience(
            self.patience_config, early_stop_key, early_stop_minimize)

        self.model = model.eval()
        self.model.requires_grad_(False)
        self.model.generation_config.cache_implementation = "static"  # Compile for generation
        logging.warning(f"WARNING: setting model.generation_config.cache_implementation=static")
        self.model.generation_config.compile_config.fullgraph = False  # Otherwise compile throws error for some models
        logging.warning(f"WARNING: setting model.generation_config.compile_config.fullgraph=False")

        self.prompt_kwargs = {}

    @torch.no_grad()
    def run(self,
            behavior_strs: List[str] | None,
            context_strs: List[str] | None,
            target_strs: List[str],
            verbose: int,
            **kwargs):

        # Local variables
        discrete_loss = None
        relaxed_loss = None
        relaxation_gap = None
        log = {
            FINAL_: {},  # Logs that will be published to wandb
            TOKEN_LVL_: {},  # To accumulate token level metrics
            STEP_LVL_: {},  # To accumulate attack step level metrics
            ATTACK_LVL_: {}  # To accumulate attack level metrics
        }

        self.prefix_cache = None  # reset at beginning of attack
        self.prefix_cache_end_id = None  # reset at beginning of attack

        # Reset stateful early stopping module
        self.esp = self.esp.reset()

        # Construct prompt manager for handling batched attack
        prompt_mgr = PromptManager(
            get_embedding_matrix(self.model), self.tokenizer, self.template, behavior_strs, context_strs, target_strs,
            self.control_prefix_length, self.control_suffix_length, self.target_length, self.control_prefix_init,
            self.control_suffix_init, **self.prompt_kwargs)
        self.print(prompt_mgr.example_prompt())

        self.disallowed_tokens = (None if self.allow_non_ascii else get_nonascii_toks(self.tokenizer))

        # Init main tensor to optimize over
        embedding_factors = self.init_embedding_factors(
            self.model, prompt_mgr)
        # Init mask to optimize over for variable length
        embedding_mask = self.maybe_init_embedding_mask(embedding_factors)
        # Init optimizer and scheduler
        optimizer = self.get_optimizer(embedding_factors, embedding_mask)
        scheduler = None
        if (self.lr_scheduler_config and 'name' in self.lr_scheduler_config and self.lr_scheduler_config['name']):
            scheduler = self.get_scheduler(optimizer, self.lr_scheduler_config)

        # Test before launching attack
        test_cases = prompt_mgr.construct_instruction_str(
            [i[m] for i, m in zip(prompt_mgr.input_ids, prompt_mgr.input_mask)])
        input_strs = [self.template.format(instruction=tc) for tc in test_cases]
        with torch.inference_mode():
            is_refusal, completions, _ = check_refusal_completions(self.model, self.tokenizer, inputs=input_strs)
        log = self.log(log, 0, input_strs, is_refusal, completions, runtime=0., verbose=verbose)

        ### Attack loop ###
        for step in range(self.num_steps):
            start = time.time()

            self.maybe_anneal(step, embedding_mask, relaxation_gap)

            optimizer.zero_grad()
            # with torch.autograd.detect_anomaly():  # TODO: only for debugging
            relaxed_loss = self.grad_fn(
                step, prompt_mgr, embedding_factors, embedding_mask, verbose=verbose, log=log)

            # Check difference in parameters being optimized (part 1)
            if verbose:
                embedding_factors_old = embedding_factors.clone()
                if embedding_mask is not None:
                    embedding_mask_old = embedding_mask.clone()

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # Check difference in parameters being optimized (part 2)
            if verbose:
                param_diff_emb = (embedding_factors - embedding_factors_old).max()
                if embedding_mask is not None:
                    param_diff_m = (embedding_mask - embedding_mask_old).max()
                    self.print(f'Max param diff {param_diff_emb:.3g} and {param_diff_m:.3g}')
                else:
                    self.print(f'Max param diff {param_diff_emb:.3g}')

            # Check if attack has diverged
            if not embedding_factors.isfinite().all() or (
                    embedding_mask is not None and not embedding_mask.isfinite().all()):
                if self.strategy != 'gbda' or embedding_factors.isnan().any():
                    logging.warning(f'Attack diverged {step=} {embedding_factors=} {embedding_mask=}')

                    # Recover from nan
                    if 'value' in self.patience_config and self.patience_config['value']:
                        reset_mask = (~embedding_factors.isfinite()).any(-1).any(-1)
                        if embedding_mask is not None:
                            reset_mask |= (~embedding_mask.isfinite()).any(-1)

                        self.reset_optimizer_for_prompt(reset_mask, optimizer, embedding_factors, embedding_mask)
                        if self.esp.has_best():
                            self.esp.patience_maybe_retrieve(step, embedding_factors, embedding_mask, reset_mask.cpu())

            if discrete_loss is not None:
                relaxation_gap = (discrete_loss['target'] - relaxed_loss['target']) / discrete_loss['target']
            entropy_factor_overwrite = self.dynamic_entropy_factor(relaxation_gap, scheduler)

            self.maybe_project(
                embedding_factors,
                embedding_mask=embedding_mask,
                entropy_factor_overwrite=entropy_factor_overwrite,
                log=log,
                verbose=verbose)

            discrete_state = self.discretize(prompt_mgr, embedding_factors, embedding_mask, log=log, verbose=verbose)
            discrete_loss, input_ids = self.discrete_loss(step, prompt_mgr, *discrete_state, log=log)

            # Early stopping and patience
            best_input_ids, best_target_strs, best_discrete_loss, _ = self.esp(
                step, embedding_factors, embedding_mask, discrete_loss,
                relaxed_loss, *discrete_state, input_ids, log)

            # Track runtime
            runtime = time.time() - start
            append_or_allocate(log[STEP_LVL_], 'runtime', np.full((prompt_mgr.n_prompts,), runtime))
            if (verbose == 2) or step % 50 == 0:
                self.print(f'Runtime of iteration {step} was {runtime} s')
                self.print(
                    f'Average runtime at iteration {step} is {np.stack(log[STEP_LVL_]['runtime'])[:, 0].mean()} s')

            test_cases = prompt_mgr.construct_instruction_str(input_ids)
            append_or_allocate(log[STEP_LVL_], 'test_cases', test_cases)

            # Generate and test
            if (step + 1) % self.eval_steps == 0 or step == self.num_steps - 1:
                if self.eval_best:
                    test_cases = prompt_mgr.construct_instruction_str(best_input_ids)
                input_strs = [self.template.format(instruction=tc) for tc in test_cases]
                with torch.inference_mode():
                    is_refusal, completions, _ = check_refusal_completions(
                        self.model, self.tokenizer, inputs=input_strs)
                log = self.log(log, step + 1, input_strs, is_refusal, completions, runtime, verbose=verbose)

                if 'forward_greedy_gens' in log[STEP_LVL_]:
                    matches = [t.startswith(f) for f, t in zip(log[STEP_LVL_]['forward_greedy_gens'][-1], completions)]
                    for pos_idx, is_match in enumerate(matches):
                        if is_match:
                            continue
                        logging.warning(
                            f"\nInconsistency! Eval generation[{pos_idx}]: {completions[pos_idx]} does not start with "
                            f"forward generation {log[STEP_LVL_]['forward_greedy_gens'][-1][pos_idx]}")

                if 'step_lvl' in log and 'loss_discrete_target_reinforce_0_reward' in log['step_lvl']:
                    greedy_rewards = torch.stack(log['step_lvl']['loss_discrete_target_reinforce_0_reward'])
                    best_greedy_step = self.esp.best_step
                    best_greedy_reward = greedy_rewards[self.esp.best_step, torch.arange(len(self.esp.best_step))]
                    logging.info(f'Best greedy step {best_greedy_step.tolist()} reward {best_greedy_reward.tolist()}')
                    if wandb.run is not None:
                        ops = [('mean', lambda x: x.float().mean()), ('max', torch.max), ('min', torch.min)]
                        wandb.log({
                            'greedy_reward_median': torch.median(best_greedy_reward).item(),
                            'greedy_reward_mean': best_greedy_reward.mean().item(),
                            'greedy_reward_min': best_greedy_reward.min().item(),
                            'greedy_reward_max': best_greedy_reward.max().item()
                        } | {
                            f'{k}_{op}': op_(vals[-1]).item()
                            for k, vals in log[STEP_LVL_].items() if 'loss_discrete' in k for op, op_ in ops
                        })

        best_instruction = prompt_mgr.construct_instruction_str(best_input_ids, best_target_strs)
        maybe_broadcast_token_lvl_logs(log, prompt_mgr.n_prompts)
        aggregate_logs(log, relaxed_loss.keys())
        plots = {}
        if verbose >= 1:
            plots = generate_wandb_plots(log)

        logging.info(torch.cuda.memory_summary())

        return best_instruction, step, log, plots

    def split_control_and_target(self, prompt_mgr: PromptManager,
                                 embedding_factors: torch.Tensor,
                                 embedding_mask: torch.Tensor | None):
        target_factors = None
        target_mask = None

        control_prefix_factors = None
        control_prefix_mask = None
        if self.control_prefix_length:
            control_prefix_factors = embedding_factors[..., :prompt_mgr.control_prefix_length, :]
            if embedding_mask is not None:
                control_prefix_mask = embedding_mask[..., :prompt_mgr.control_prefix_length]

        control_suffix_factors = None
        control_suffix_mask = None
        if self.control_suffix_length:
            # To save memory in the "default" case
            if prompt_mgr.control_suffix_length == embedding_factors.shape[1]:
                control_suffix_factors = embedding_factors
                control_suffix_mask = embedding_mask
            else:
                control_suffix_factors = embedding_factors[..., -prompt_mgr.control_suffix_length:, :]
                if embedding_mask is not None:
                    control_suffix_mask = embedding_mask[..., -prompt_mgr.control_suffix_length:]

        return (control_prefix_factors, control_prefix_mask,
                control_suffix_factors, control_suffix_mask,
                target_factors, target_mask)

    def init_controls_and_targets(self, target_strs):
        control_strs = ' '.join(self.control_length * ['!'])
        control_strs = len(target_strs) * [control_strs]

        return control_strs, target_strs

    def anneal_step(self, step: int):
        def weighting_uniform(step: int, init_value: float, end_value: float, duration: int) -> float:
            return init_value + (end_value - init_value) * (min(step, duration) / duration)

        if self.anneal_config['mode'] == 'uniform':
            weighting = weighting_uniform
        elif self.anneal_config['mode'] == 'uniform_log':
            def weighting(step: int, init_value: float, end_value: float, duration: int) -> float:
                return init_value * (end_value / init_value)**(min(step, duration) / duration)
        elif self.anneal_config['mode'] == 'lin_order':

            def weighting(step: int, init_value: float, end_value: float, duration: int) -> torch.Tensor:
                uniform = weighting_uniform(step, init_value, end_value, duration)
                # TODO: make number 5 configurable
                lin_seq = uniform * torch.linspace(5, 1, self.control_length)
                return torch.clamp_max(lin_seq, end_value)[None, :]
        else:
            raise ValueError(
                f"Anneal mode {self.anneal_config['mode']} not supported")

        for attr in self.anneal_config['attrs']:
            if 'gbda' in attr:
                continue
            init_value = self.anneal_config.get(f'init_{attr}', 0)
            end_value = self.anneal_config.get(f'end_{attr}', 1)
            step_ = step - self.anneal_config.get(f'start_{attr}', self.anneal_config['start'])
            step_ = 0 if step_ < 0 else step_
            duration = self.anneal_config.get(f'duration_{attr}', self.anneal_config['duration'])
            setattr(self, attr, weighting(0 if step_ < 0 else step_, init_value, end_value, duration))

        if self.strategy == 'gbda':
            if 'gbda_temp' in self.anneal_config['attrs']:
                self.gbda_config['temp'] = weighting(
                    step, self.gbda_config['anneal_init_temp'], self.gbda_config['anneal_end_temp'],
                    self.anneal_config['duration'])
            gumble_hard_from = self.gbda_config['soft_frac'] * self.anneal_config['duration']
            enable_hard_gumble = self.gbda_config['soft_frac'] and step >= gumble_hard_from
            if enable_hard_gumble:
                self.gbda_config['gumble_hard'] = True

    def maybe_anneal(self, step: int, embedding_mask: torch.Tensor | None, relaxation_gap: torch.Tensor | None):
        if self.anneal and step >= 0:
            self.anneal_step(step)
        self.maybe_anneal_embedding_mask(embedding_mask, relaxation_gap)

    def maybe_anneal_embedding_mask(self, embedding_mask: torch.Tensor | None, relaxation_gap: torch.Tensor | None):
        if not self.anneal or embedding_mask is None or relaxation_gap is None:
            return
        max_at = self.variable_control_length['max_anneal_temperature']
        if max_at == 1:
            return
        relaxation_gap = torch.clamp(relaxation_gap, 0, 1)
        control_k = (max_at - 1) * relaxation_gap + 1
        embedding_mask_ = x_bounded_sigmoid(embedding_mask, k=control_k[..., None].to(embedding_mask.device))
        embedding_mask_ = embedding_mask_.clamp(self.eps, 1)
        embedding_mask.data.copy_(embedding_mask_)
        return

    def init_embedding_factors(self, model, prompt_mgr: PromptManager):
        n_token_dim = model.get_input_embeddings().weight.shape[0]

        n_prompts = prompt_mgr.n_prompts

        c_length = (prompt_mgr.control_prefix_length
                    + prompt_mgr.control_suffix_length)
        shape = (n_prompts, c_length, n_token_dim)
        if self.initialization == 'control':
            input_ids = torch.cat(
                (prompt_mgr.input_ids[prompt_mgr.c_prefix_batch_id, prompt_mgr.c_prefix_tok_id],
                 prompt_mgr.input_ids[prompt_mgr.c_suffix_batch_id, prompt_mgr.c_suffix_tok_id]), dim=-1)
            assert input_ids.shape[1] == c_length
            embedding_factors = one_hot_tensor(
                input_ids.to(model.device),
                n_token_dim, model.device, torch.float32)
        else:
            embedding_factors = torch.rand(shape, dtype=torch.float32)
            embedding_factors = embedding_factors.to(model.device)

        if self.disallowed_tokens is not None:
            embedding_factors[..., self.disallowed_tokens] = -np.infty if self.strategy == 'gbda' else 0.

        self.eps = torch.finfo(embedding_factors.dtype).eps
        embedding_factors = embedding_factors / torch.clamp_min(embedding_factors.sum(-1, keepdims=True), self.eps)

        return embedding_factors

    def maybe_init_embedding_mask(self, embedding_factors: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.variable_control_length['enable']:
            return None
        embedding_mask = torch.rand(
            embedding_factors.shape[:-1], dtype=embedding_factors.dtype, device=embedding_factors.device)

        embedding_mask = torch.clamp(embedding_mask, self.eps, 1)
        return embedding_mask

    def prepare_embedding_factors(self, prompt_mgr: PromptManager, embedding_factors: torch.Tensor) -> torch.Tensor:
        if self.strategy == 'gbda' and self.gbda_config['temp'] >= 0:
            if self.gbda_config['deterministic']:
                return F.softmax(embedding_factors / self.gbda_config['temp'], -1)
            else:
                return F.gumbel_softmax(
                    embedding_factors, tau=self.gbda_config['temp'], hard=self.gbda_config['gumble_hard'], dim=-1)
        else:
            embedding_factors = embedding_factors / torch.clamp_min(embedding_factors.sum(-1, keepdims=True), self.eps)
        return embedding_factors

    def prepare_embedding_mask(
            self, prompt_mgr: PromptManager, embedding_mask: torch.Tensor | None) -> torch.Tensor | None:
        if embedding_mask is None:
            return embedding_mask

    def randomize_embedding_factors(self, embedding_factors: torch.Tensor):
        noise = torch.randn_like(embedding_factors)
        noise *= self.langevin_dynamics_std

        embedding_factors.add_(noise)

    def discretize_embedding_mask(
            self, embedding_mask: torch.Tensor) -> torch.Tensor:
        embedding_mask_max = embedding_mask.max(-1, keepdim=True).values
        return ((embedding_mask >= 0.5) |
                (embedding_mask == embedding_mask_max))

    def _discretize(self,
                    embedding_factors: torch.Tensor | None,
                    embedding_mask: torch.Tensor | None,
                    prompt_mgr: PromptManager,
                    extend_length: bool = False,
                    verbose: int = 0,
                    log: dict | None = None
                    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        idx = embedding_factors.argmax(-1)

        if embedding_mask is not None:
            mask = self.discretize_embedding_mask(embedding_mask)
        else:
            mask = torch.ones_like(idx, dtype=bool)

        strings = self.tokenizer.batch_decode([i[m] for i, m in zip(idx, mask)], clean_up_tokenization_spaces=False)

        # Below we decode to string, encode again, and handle length mismatches
        idx_retok = self.tokenizer(strings, add_special_tokens=False, padding=False)['input_ids']
        max_len = max([len(c) for c in idx_retok])

        if verbose:
            len_diff = [m.sum().item() - len(n) for m, n in zip(mask, idx_retok)]

            self.print('Discretized len diff', len_diff)
            if log is not None:
                len_diff_log = torch.tensor(len_diff).broadcast_to(
                    (prompt_mgr.n_prompts,))
                append_or_allocate(log[STEP_LVL_], 'len_diff_decode_encode', len_diff_log)

        if max_len > idx.shape[-1] and extend_length:
            idx = torch.zeros((idx.shape[0], max_len), dtype=idx.dtype, device=idx.device)
            mask = torch.zeros_like(idx, dtype=bool)

        for i, idx_i in enumerate(idx_retok):
            idx[i, :len(idx_i)] = torch.tensor(idx_i[:idx.shape[-1]])
            mask[i, :len(idx_i)] = True
            mask[i, len(idx_i):] = False

        if not extend_length:
            strings = self.tokenizer.batch_decode([i[m] for i, m in zip(idx, mask)], clean_up_tokenization_spaces=False)

        return idx, mask, strings

    def discretize(self,
                   prompt_mgr: PromptManager,
                   embedding_factors: torch.Tensor,
                   embedding_mask: torch.Tensor | None = None,
                   *args,
                   **kwargs):
        out = self.split_control_and_target(
            prompt_mgr, embedding_factors, embedding_mask)
        out = [(None, None, None) if f is None else self._discretize(f, m, prompt_mgr, *args, **kwargs)
               for f, m in list(zip(out[::2], out[1::2]))]
        # Flatten out
        out = [e for o in out for e in o]
        return out

    def maybe_project(self,
                      embedding_factors: torch.Tensor,
                      embedding_mask: torch.Tensor | None,
                      entropy_factor_overwrite: float | torch.Tensor
                      | None = None,
                      log: dict | None = None,
                      verbose: int = 1):
        entropy_factor = self.entropy_factor
        if entropy_factor_overwrite is not None:
            entropy_factor = entropy_factor_overwrite

        if isinstance(entropy_factor, torch.Tensor):
            entropy_factor = entropy_factor.to(embedding_factors.device, non_blocking=True)
            entropy_factor = torch.clamp(entropy_factor, 0, 1)
        else:
            entropy_factor = max(0, min(entropy_factor, 1))

        if embedding_mask is not None:
            embedding_mask.data.clamp_(self.eps, 1)

            scale_entf = self.variable_control_length['scale_entropy_factor']
            if scale_entf == 'sigmoid':
                entropy_factor = (entropy_factor * x_bounded_sigmoid(embedding_mask))
            elif scale_entf == 'lin':
                entropy_factor = entropy_factor * embedding_mask

            if verbose:
                if log is not None:
                    append_or_allocate(log[TOKEN_LVL_], 'embedding_mask', embedding_mask)
                    append_or_allocate(log[TOKEN_LVL_], 'entropy_factor', entropy_factor)
                if verbose > 1:
                    self.print('embedding_mask', [[f'{v:.3g}' for v in row] for row in embedding_mask.cpu()])
                    self.print('entropy_factor', [[f'{v:.3g}' for v in row] for row in entropy_factor.cpu()])

        if self.strategy == 'gbda':
            do_project = False
            if isinstance(entropy_factor, torch.Tensor):
                if (entropy_factor > 0).any():
                    do_project = True
            else:
                do_project = entropy_factor > 0
            if do_project:
                log_probs = torch.log_softmax(embedding_factors, -1)
                logit_offset = (log_probs.max(-1).values - embedding_factors.max(-1).values)
                probs = log_probs.exp()
                probs_ = self.tsallis_q2_projection(probs, entropy_factor)
                embedding_factors_ = probs_.log() - logit_offset[..., None]
                embedding_factors.data.copy_(embedding_factors_)
            return embedding_factors

        if verbose:
            self.projection_logs(embedding_factors, 'simplex', log=log, prefix='pre_proj', verbose=verbose)

        embedding_factors_ = self.simplex_projection(embedding_factors)
        embedding_factors.data.copy_(embedding_factors_)

        if self.langevin_dynamics_std:
            self.randomize_embedding_factors(embedding_factors)

            embedding_factors_ = self.simplex_projection(embedding_factors)
            embedding_factors.data.copy_(embedding_factors_)

        if verbose:
            self.projection_logs(
                embedding_factors, ('simplex', 'entropy'), log=log, prefix='pre_ent_proj', verbose=verbose)

        if (entropy_factor.any() if isinstance(entropy_factor, torch.Tensor) else entropy_factor):
            embedding_factors_ = self.tsallis_q2_projection(embedding_factors, entropy_factor)
            embedding_factors.data.copy_(embedding_factors_)

        if verbose:
            self.projection_logs(
                embedding_factors, ('simplex', 'entropy'), log=log, prefix='post_proj', verbose=verbose)

        return

    def simplex_projection(self, values, **kwargs):
        values = values.clone()
        exceeds_budget = torch.clamp(values, 0, 1).sum(-1) > 1
        if exceeds_budget.any():
            if 'bisection' in self.simplex_proj_method:
                left = (values - 1).min(-1).values
                right = values.max(-1).values
                miu = self.simplex_bisection(values, left, right)
                miu = torch.where(exceeds_budget, miu, 0)
                values = torch.clamp(values - miu[..., None], min=0, max=1)
            else:
                values[exceeds_budget] = self.simplex_sort_projection(
                    values[exceeds_budget])
                values[~exceeds_budget] = torch.clamp(values[~exceeds_budget], min=0, max=1)
        else:
            values = torch.clamp(values, min=0, max=1)

        # Handle degenerate case where weights for token are all 0
        all_values_zero_offset = (
            torch.isclose(values.sum(-1, keepdims=True), torch.tensor(0.)) *
            torch.rand_like(values))
        values += all_values_zero_offset
        values = values / torch.clamp_min(values.sum(-1, keepdims=True), self.eps)

        return values

    @staticmethod
    def simplex_bisection(values, a, b, epsilon=1e-5, iter_max=25):

        def func(x):
            return torch.clamp(values - x[..., None], 0, 1).sum(-1) - 1

        miu = a
        for _ in range(int(iter_max)):
            miu = (a + b) / 2
            # Decide the side to repeat the steps
            crit = func(miu) * func(a)
            b = torch.where(crit < 0, miu, b)
            a = torch.where(crit > 0, miu, a)
            if ((b - a) <= epsilon).all():
                break
        return miu

    @staticmethod
    def simplex_sort_projection(values: torch.Tensor) -> torch.Tensor:
        """Projects the values back into the simplex based on "Large-scale
        Multiclass Support Vector Machine Training via Euclidean Projection
        onto the Simplex Mathieu Blondel, Akinori Fujino, and Naonori Ueda.
        ICPR 2014."

        Args:
            values (torch.Tensor): of shape batch b x dimensions d.

        Returns:
            torch.Tensor: of the values projected onto the simplex.
        """
        b, d = values.shape
        cat_indices = torch.arange(d, device=values.device)
        batch_indices = torch.arange(b, device=values.device)

        values = torch.clamp_min(values, 0.)

        values_sorted = -(-values).sort(-1).values
        values_cumulative = torch.cumsum(values_sorted, axis=-1) - 1
        condition = values_sorted - values_cumulative / (cat_indices + 1) > 0
        rho = torch.count_nonzero(condition, axis=-1)
        theta = values_cumulative[batch_indices, rho - 1] / rho
        values = torch.clamp_min(values - theta[:, np.newaxis], 0.)
        return values

    def tsallis_q2_projection(self, values: torch.Tensor, entropy_factor: float | torch.Tensor) -> torch.Tensor:
        """Entropy factor within (0,1] that scales between max and min."""
        assert isinstance(self.tsallis_q2_proj_config['iter'], int)

        normal = torch.ones((values.shape[-1], ), device=values.device)
        normal[self.disallowed_tokens] = 0

        for _ in range(self.tsallis_q2_proj_config['iter']):
            if self.tsallis_q2_proj_config['exclude_already_zero']:
                is_close_to_zero = torch.isclose(values, torch.tensor(0.))
                normal = torch.broadcast_to(normal[None], is_close_to_zero.shape).clone()
                normal[is_close_to_zero] = 0
                normal = normal / normal.norm(dim=-1, keepdim=True)
            else:
                normal = normal / normal.norm()

            non_zero_components = normal > 0
            d = non_zero_components.sum(-1)
            target_entropy = (1 - entropy_factor) * (d - 1) / d
            center = 1 / d[..., None] * non_zero_components

            dist_to_hyperplane = (values * normal).sum(-1)
            projection_radius = torch.sqrt(torch.clamp(1 - target_entropy - dist_to_hyperplane**2, 0))[..., None]

            direction = values - center
            direction_norm = torch.linalg.norm(direction, axis=-1, keepdims=True)
            direction_norm = torch.clamp_min(direction_norm, self.eps)
            exceeds_budget = (direction_norm < projection_radius)[..., 0]

            if not exceeds_budget.any():
                break

            values_ = projection_radius / direction_norm * direction + center
            # TODO: try more advanced strategies
            values_[exceeds_budget] = self.simplex_projection(
                values_[exceeds_budget])
            values = torch.where(exceeds_budget[..., None], values_, values)

        return values

    def get_optimizer(self, embedding_factors: torch.Tensor, embedding_mask: torch.Tensor) -> torch.optim.Optimizer:
        optimizer_cls = getattr(torch.optim, self.optimizer_config['name'])
        optimizer_config = {
            k: v for k, v in self.optimizer_config.items() if k != 'name' and k != 'param_groups'}
        if self.variable_control_length['enable']:
            params = [
                {
                    'params': [embedding_factors],
                    **self.optimizer_config['param_groups']['embedding_factors']
                },
                {
                    'params': [embedding_mask],
                    **self.optimizer_config['param_groups']['embedding_mask']
                },
            ]
        else:
            params = [embedding_factors]
        optimizer = optimizer_cls(params, self.learning_rate,
                                  **optimizer_config)
        return optimizer

    def get_scheduler(self, optimizer: torch.optim.Optimizer, config: dict) -> torch.optim.lr_scheduler.LRScheduler:
        match config['name']:
            case 'ChainedScheduler':
                return torch.optim.lr_scheduler.ChainedScheduler(
                    [self.get_scheduler(optimizer, cfg) for cfg in config['schedulers']])
            case 'SequentialLR':
                schedulers = [self.get_scheduler(optimizer, cfg) for cfg in config['schedulers']]
                return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers, config['milestones'])
            case 'LogarithmicLR':
                scheduler_cls = LogarithmicLR
            case _:
                scheduler_cls = getattr(torch.optim.lr_scheduler, config['name'])

        lr_scheduler_config = {
            k: v
            for k, v in config.items() if k != 'name'
        }
        scheduler = scheduler_cls(optimizer, **lr_scheduler_config)
        return scheduler

    def dynamic_entropy_factor(
        self,
        relaxation_gap: torch.Tensor | None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    ) -> float | torch.Tensor:
        entropy_factor_overwrite = None
        if (self.entropy_factor_scale_by_relaxation_gap
                and relaxation_gap is not None):
            relaxation_gap_scale = relaxation_gap.clamp(0, 1)
            squeeze = 1 / (1 - self.entropy_factor_scale_by_relaxation_gap)
            relaxation_gap_scale = torch.where(
                relaxation_gap_scale
                < self.entropy_factor_scale_by_relaxation_gap,
                x_bounded_sigmoid(squeeze * relaxation_gap_scale), 1)
            relaxation_gap_scale = relaxation_gap_scale[:, None]

            entropy_factor_overwrite = (relaxation_gap_scale *
                                        self.entropy_factor)

        if self.entropy_factor_alternate_scheduler and scheduler is not None:
            if hasattr(scheduler, '_schedulers'):
                base_lr = scheduler._schedulers[0].base_lrs[0]
                # To allow reverting the annealing (eta_min > learning rate)
                if hasattr(scheduler._schedulers[-1], 'eta_min'):
                    base_lr = max(base_lr, scheduler._schedulers[-1].eta_min)
            else:
                base_lr = scheduler.base_lrs[0]
                # To allow reverting the annealing (eta_min > learning rate)
                if hasattr(scheduler, 'eta_min'):
                    base_lr = max(base_lr, scheduler.eta_min)
            last_lr = scheduler.get_last_lr()[0]
            if entropy_factor_overwrite is None:
                entropy_factor_overwrite = self.entropy_factor
            entropy_factor_overwrite *= last_lr / base_lr

        return entropy_factor_overwrite

    @torch.enable_grad()
    def _grad_fn(self,
                 step_idx: int,
                 prompt_mgr: PromptManager,
                 embedding_factors: torch.Tensor,
                 embedding_mask: torch.Tensor | None,
                 log: dict = None,
                 verbose: int = 1) -> Dict[str, torch.Tensor]:
        # logging.info(f'Gradient calculation')  # TODO: remove
        embedding_factors.requires_grad_()
        if embedding_mask is not None:
            embedding_mask.requires_grad_()

        embedding_factors_ = self.prepare_embedding_factors(prompt_mgr, embedding_factors)
        embedding_mask_ = self.prepare_embedding_mask(prompt_mgr, embedding_mask)

        control_and_target = self.split_control_and_target(prompt_mgr, embedding_factors_, embedding_mask_)

        inputs_embeds, mask = prompt_mgr.soft_prompt(*control_and_target)

        logits = self.forward_maybe_prefix_cache(
            prompt_mgr, dict(inputs_embeds=inputs_embeds, attention_mask=mask))

        loss_dict = self.loss(
            logits,
            prompt_mgr,
            control_and_target,
            log=log,
            logver_prefix='relaxed')

        loss = loss_dict['combined'].sum()
        loss.backward()

        return loss_dict

    def grad_fn(self,
                step_idx: int,
                prompt_mgr: PromptManager,
                embedding_factors: torch.Tensor,
                embedding_mask: torch.Tensor | None,
                log: dict = None,
                verbose: int = 1) -> Tuple[torch.Tensor, float]:
        loss_dict = self._grad_fn(step_idx, prompt_mgr, embedding_factors, embedding_mask, log, verbose)

        loss_dict = {k: v.detach().float().to('cpu', non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in loss_dict.items()}

        factors_grad = embedding_factors.grad

        if self.disallowed_tokens is not None:
            factors_grad[..., self.disallowed_tokens] = 0.

        if verbose and log is not None:
            append_or_allocate(log[TOKEN_LVL_], 'grad_raw_max', factors_grad.max(-1).values)
            append_or_allocate(log[TOKEN_LVL_], 'grad_raw_min', factors_grad.min(-1).values)
            append_or_allocate(log[TOKEN_LVL_], 'grad_raw_norm', torch.linalg.norm(factors_grad, axis=-1))

        self._maybe_clip_gradient(factors_grad, self.grad_clip_strategy)

        if embedding_mask is not None and embedding_mask.grad is not None:
            mask_grad = embedding_mask.grad
            if verbose and log is not None:
                append_or_allocate(log[TOKEN_LVL_], 'mask_grad_raw', mask_grad)
            self._maybe_clip_gradient(mask_grad, 'value')

        return loss_dict

    def _maybe_clip_gradient(self, grad: torch.Tensor,
                             grad_clip_strategy: str) -> torch.Tensor:
        if grad_clip_strategy == 'norm':
            norm = torch.linalg.norm(grad)
            if norm > self.grad_clip_value:
                grad *= self.grad_clip_value / (norm + self.eps)
        elif grad_clip_strategy == 'token_norm':
            norm = torch.linalg.norm(grad, axis=-1, keepdim=True)
            grad_ = torch.where(
                norm > self.grad_clip_value,
                self.grad_clip_value * grad / (norm + self.eps),
                grad
            )
            grad.copy_(grad_)
        elif grad_clip_strategy == 'value':
            grad.clamp_(-self.grad_clip_value, self.grad_clip_value)
        return grad

    def discrete_loss(self,
                      step_idx: int,
                      prompt_mgr: PromptManager,
                      control_prefix_ids: torch.Tensor | None,
                      control_prefix_mask: torch.Tensor | None,
                      control_prefix_str: str | None,
                      control_suffix_ids: torch.Tensor | None,
                      control_suffix_mask: torch.Tensor | None,
                      control_suffix_str: str | None,
                      target_ids: torch.Tensor | None,
                      target_mask: torch.Tensor | None,
                      target_str: str | None,
                      log: dict | None = None) -> Tuple[Dict[str, float], List[torch.Tensor]]:
        control_and_target = (
            control_prefix_ids, control_prefix_mask,
            control_suffix_ids, control_suffix_mask,
            target_ids, target_mask
        )
        input_ids, attention_mask = prompt_mgr.hard_prompt(*control_and_target)
        logits = self.forward_maybe_prefix_cache(prompt_mgr, dict(input_ids=input_ids, attention_mask=attention_mask))

        control_and_target = (
            None, control_prefix_mask,
            None, control_suffix_mask,
            None, target_mask
        )
        loss_dict = self.loss(logits,
                              prompt_mgr,
                              control_and_target,
                              input_ids,
                              attention_mask,
                              log=log,
                              logver_prefix='discrete')

        loss_dict = {k: v.detach().float().to('cpu', non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in loss_dict.items()}

        input_ids = [i[m.bool()].to('cpu', non_blocking=True) for i, m in zip(input_ids, attention_mask)]

        return loss_dict, input_ids

    def log(self,
            log,
            step_num,
            input_strs,
            is_refusal,
            completions,
            runtime,
            verbose=True):
        n_passed = len(is_refusal) - sum(is_refusal)

        log[FINAL_]['runtime'] = runtime
        log[FINAL_]['tests'] = dict(input_strs=input_strs, completions=completions)
        log[FINAL_]['passed'] = [not r for r in is_refusal]
        log[FINAL_]['n_prompts'] = len(input_strs)

        append_or_allocate(log[STEP_LVL_], 'input_strs', input_strs)
        append_or_allocate(log[STEP_LVL_], 'generated_string', completions)
        append_or_allocate(log[STEP_LVL_], 'passed', log[FINAL_]['passed'])

        output_str = (f'Passed {n_passed:>3}/{len(input_strs):<3} \n')
        self.print(f"\n====================================================\n"
                   f"Step {step_num:>4}/{self.num_steps:>4} ({runtime:.4} s)\n"
                   f"{output_str}"
                   #    f"control='{controls}'\n"input_strs
                   #    f"input_strs='{input_strs}'\n"
                   f"====================================================\n")
        return log

    def projection_logs(self,
                        values: torch.Tensor,
                        which: Tuple[str, ...],
                        log: dict | None = None,
                        prefix='pre_proj',
                        verbose: bool = False):

        statistics = {}
        if 'simplex' in which:
            statistics.update({
                'wkt_sum':
                lambda val: torch.clamp(val, 0, 1).sum(-1),
                'wkt_max':
                lambda val: torch.clamp(val, 0, 1).max(-1).values,
                'wkt_nnz':
                lambda val: (torch.clamp(val, 0, 1) > 0).sum(-1)
            })
        if 'entropy' in which:
            statistics.update({'entropy': lambda val: tsallis(val, q=2)})

        for statistic, func in statistics.items():
            statistic_values = func(values)
            if log is not None:
                append_or_allocate(log[TOKEN_LVL_], f'{prefix}_{statistic}',
                                   statistic_values)
            if verbose:
                self.print(prefix, statistic, [[f'{v:.3g}' for v in row]
                                               for row in statistic_values])

    def reset_optimizer_for_prompt(self, reset_mask: torch.Tensor, optimizer: torch.optim.Optimizer,
                                   embedding_factors: torch.Tensor, embedding_mask: torch.Tensor | None):
        for key, value in optimizer.state[embedding_factors].items():
            if key == 'step':
                continue
            value[reset_mask & (~value.isfinite()).any(-1).any(-1)] = 0.
        if embedding_mask is not None:
            for key, value in optimizer.state[embedding_mask].items():
                if key == 'step':
                    continue
                value[reset_mask & (~value.isfinite()).any(-1)] = 0.

    def print(self, *args, fn=logging.info):
        out = ' '.join(map(str, args))
        fn(out)

    def forward_maybe_prefix_cache(
            self, prompt_mgr: PromptManager, model_input: Dict[str, torch.Tensor], **model_kwargs) -> torch.Tensor:
        if not self.use_prefix_cache or not torch.is_grad_enabled():
            # Skipping when grad is disabled is due to the differing length of tokenizer.encode(tokenizer.decode(x))
            # since it requires setting an attention mask with the current static "template" of the prompt manager
            return self.model(**model_input, use_cache=False, **model_kwargs).logits

        assert not model_kwargs, 'No kwargs supported with caching but received {model_kwargs}'

        if 'attention_mask' in model_input:
            assert model_input['attention_mask'].all(), 'No token must be masked out for caching'
            del model_input['attention_mask']

        orig_use_cache = self.model.config.use_cache
        self.model.config.use_cache = self.use_prefix_cache

        with torch.no_grad():
            if self.prefix_cache is None:
                if prompt_mgr.control_prefix_length:
                    self.prefix_cache_end_id = prompt_mgr.c_prefix_tok_id[:, 0].min().item()
                else:
                    self.prefix_cache_end_id = prompt_mgr.c_suffix_tok_id[:, 0].min().item()
                model_input_ = {k: v for k, v in model_input.items()}
                for field in model_input_.keys():
                    model_input_[field] = model_input_[field][:, :self.prefix_cache_end_id]
                self.prefix_cache = self.model(**model_input_, use_cache=True, **model_kwargs).past_key_values
        for field in model_input.keys():
            n_token = model_input[field].shape[1]
            model_input[field] = model_input[field][:, self.prefix_cache_end_id:]
        model_input['past_key_values'] = self.prefix_cache

        logits = self.model(**model_input).logits
        logits_ = torch.zeros((logits.shape[0], n_token, logits.shape[2]), device=logits.device, dtype=logits.dtype)
        logits_[:, -logits.shape[1]:] = logits

        self.model.config.use_cache = orig_use_cache
        return logits_
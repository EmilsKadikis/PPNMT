import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn


from transformers.pytorch_utils import torch_int_div
from transformers.utils import ModelOutput, logging
from transformers import DisjunctiveConstraint, PhrasalConstraint
from transformers import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers import GenerationConfig
from transformers import (
    LogitsProcessorList,
)
from transformers import (
    StoppingCriteriaList,
    GenerationMixin
)

logger = logging.get_logger(__name__)

class PerturbableGenerationMixin(GenerationMixin):

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = False,
        **kwargs,
    ): #-> Union[GenerateOutput, torch.LongTensor]:
        r"""

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            kwargs:
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GreedySearchDecoderOnlyOutput`],
                    - [`~generation.SampleDecoderOnlyOutput`],
                    - [`~generation.BeamSearchDecoderOnlyOutput`],
                    - [`~generation.BeamSampleDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GreedySearchEncoderDecoderOutput`],
                    - [`~generation.SampleEncoderDecoderOutput`],
                    - [`~generation.BeamSearchEncoderDecoderOutput`],
                    - [`~generation.BeamSampleEncoderDecoderOutput`]
        """
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()

        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation -- update the generation config
            # model attribute accordingly, if it was created from the model config
            if self.generation_config._from_model_config:
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed soon, in a future version."
                        " Please use a generation configuration file (see"
                        " https://huggingface.co/docs/transformers/main_classes/text_generation)"
                    )
                    self.generation_config = new_generation_config
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        # 3. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        # 4. Define other model kwargs
        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        model_kwargs["use_cache"] = generation_config.use_cache

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        # decoder-only models should use left-padding for generation
        if not self.config.is_encoder_decoder:
            if (
                generation_config.pad_token_id is not None
                and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids = self._prepare_decoder_input_ids_for_generation(
                batch_size,
                decoder_start_token_id=generation_config.decoder_start_token_id,
                bos_token_id=generation_config.bos_token_id,
                model_kwargs=model_kwargs,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_seq_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
                "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
            if not has_default_max_length:
                logger.warn(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)",
                    UserWarning,
                )

        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            raise ValueError(
                f"Unfeasible length constraints: the minimum length ({generation_config.min_length}) is larger than"
                f" the maximum length ({generation_config.max_length})"
            )
        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )

        # 7. determine generation mode
        is_constraint_gen_mode = (
            generation_config.constraints is not None or generation_config.force_words_ids is not None
        )

        is_contrastive_search_gen_mode = (
            generation_config.top_k is not None
            and generation_config.top_k > 1
            and generation_config.do_sample is False
            and generation_config.penalty_alpha is not None
            and generation_config.penalty_alpha > 0
        )

        is_greedy_gen_mode = (
            (generation_config.num_beams == 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is False
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_sample_gen_mode = (
            (generation_config.num_beams == 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is True
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_beam_gen_mode = (
            (generation_config.num_beams > 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is False
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_beam_sample_gen_mode = (
            (generation_config.num_beams > 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is True
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_group_beam_gen_mode = (
            (generation_config.num_beams > 1)
            and (generation_config.num_beam_groups > 1)
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )

        if generation_config.num_beam_groups > generation_config.num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and generation_config.do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )

        # 9. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        # 10. go into different generation modes
        if is_greedy_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {generation_config.num_return_sequences} when doing"
                    " greedy search."
                )

            # 11. run greedy search
            return self.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_contrastive_search_gen_mode:
            raise NotImplementedError("Contrastive search is not available for perturbations.")
        elif is_sample_gen_mode:
            raise NotImplementedError("Sampling is not yet implemented.")

            # # 11. prepare logits warper
            # logits_warper = self._get_logits_warper(generation_config)

            # # 12. expand input_ids with `num_return_sequences` additional sequences per batch
            # input_ids, model_kwargs = self._expand_inputs_for_generation(
            #     input_ids=input_ids,
            #     expand_size=generation_config.num_return_sequences,
            #     is_encoder_decoder=self.config.is_encoder_decoder,
            #     **model_kwargs,
            # )

            # # 13. run sample
            # return self.sample(
            #     input_ids,
            #     logits_processor=logits_processor,
            #     logits_warper=logits_warper,
            #     stopping_criteria=stopping_criteria,
            #     pad_token_id=generation_config.pad_token_id,
            #     eos_token_id=generation_config.eos_token_id,
            #     output_scores=generation_config.output_scores,
            #     return_dict_in_generate=generation_config.return_dict_in_generate,
            #     synced_gpus=synced_gpus,
            #     **model_kwargs,
            # )

        elif is_beam_gen_mode:
            raise NotImplementedError("Beam search is not yet implemented.")
            # if generation_config.num_return_sequences > generation_config.num_beams:
            #     raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            # if stopping_criteria.max_length is None:
            #     raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            # # 11. prepare beam search scorer
            # beam_scorer = BeamSearchScorer(
            #     batch_size=batch_size,
            #     num_beams=generation_config.num_beams,
            #     device=inputs_tensor.device,
            #     length_penalty=generation_config.length_penalty,
            #     do_early_stopping=generation_config.early_stopping,
            #     num_beam_hyps_to_keep=generation_config.num_return_sequences,
            #     max_length=generation_config.max_length,
            # )
            # # 12. interleave input_ids with `num_beams` additional sequences per batch
            # input_ids, model_kwargs = self._expand_inputs_for_generation(
            #     input_ids=input_ids,
            #     expand_size=generation_config.num_beams,
            #     is_encoder_decoder=self.config.is_encoder_decoder,
            #     **model_kwargs,
            # )
            # # 13. run beam search
            # return self.beam_search(
            #     input_ids,
            #     beam_scorer,
            #     logits_processor=logits_processor,
            #     stopping_criteria=stopping_criteria,
            #     pad_token_id=generation_config.pad_token_id,
            #     eos_token_id=generation_config.eos_token_id,
            #     output_scores=generation_config.output_scores,
            #     return_dict_in_generate=generation_config.return_dict_in_generate,
            #     synced_gpus=synced_gpus,
            #     **model_kwargs,
            # )

        elif is_beam_sample_gen_mode:
            raise NotImplementedError("Beam sampling is not yet implemented.")
            # # 11. prepare logits warper
            # logits_warper = self._get_logits_warper(generation_config)

            # if stopping_criteria.max_length is None:
            #     raise ValueError("`max_length` needs to be a stopping_criteria for now.")
            # # 12. prepare beam search scorer
            # beam_scorer = BeamSearchScorer(
            #     batch_size=batch_size * generation_config.num_return_sequences,
            #     num_beams=generation_config.num_beams,
            #     device=inputs_tensor.device,
            #     length_penalty=generation_config.length_penalty,
            #     do_early_stopping=generation_config.early_stopping,
            #     max_length=generation_config.max_length,
            # )

            # # 13. interleave input_ids with `num_beams` additional sequences per batch
            # input_ids, model_kwargs = self._expand_inputs_for_generation(
            #     input_ids=input_ids,
            #     expand_size=generation_config.num_beams * generation_config.num_return_sequences,
            #     is_encoder_decoder=self.config.is_encoder_decoder,
            #     **model_kwargs,
            # )

            # # 14. run beam sample
            # return self.beam_sample(
            #     input_ids,
            #     beam_scorer,
            #     logits_processor=logits_processor,
            #     logits_warper=logits_warper,
            #     stopping_criteria=stopping_criteria,
            #     pad_token_id=generation_config.pad_token_id,
            #     eos_token_id=generation_config.eos_token_id,
            #     output_scores=generation_config.output_scores,
            #     return_dict_in_generate=generation_config.return_dict_in_generate,
            #     synced_gpus=synced_gpus,
            #     **model_kwargs,
            # )

        elif is_group_beam_gen_mode:
            raise NotImplementedError("Group beam search is not yet implemented.")
            # if generation_config.num_return_sequences > generation_config.num_beams:
            #     raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            # if generation_config.num_beams % generation_config.num_beam_groups != 0:
            #     raise ValueError("`num_beams` should be divisible by `num_beam_groups` for group beam search.")

            # if stopping_criteria.max_length is None:
            #     raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            # has_default_typical_p = kwargs.get("typical_p") is None and generation_config.typical_p == 1.0
            # if not has_default_typical_p:
            #     raise ValueError("Decoder argument `typical_p` is not supported with beam groups.")

            # # 11. prepare beam search scorer
            # beam_scorer = BeamSearchScorer(
            #     batch_size=batch_size,
            #     num_beams=generation_config.num_beams,
            #     device=inputs_tensor.device,
            #     length_penalty=generation_config.length_penalty,
            #     do_early_stopping=generation_config.early_stopping,
            #     num_beam_hyps_to_keep=generation_config.num_return_sequences,
            #     num_beam_groups=generation_config.num_beam_groups,
            #     max_length=generation_config.max_length,
            # )
            # # 12. interleave input_ids with `num_beams` additional sequences per batch
            # input_ids, model_kwargs = self._expand_inputs_for_generation(
            #     input_ids=input_ids,
            #     expand_size=generation_config.num_beams,
            #     is_encoder_decoder=self.config.is_encoder_decoder,
            #     **model_kwargs,
            # )
            # # 13. run beam search
            # return self.group_beam_search(
            #     input_ids,
            #     beam_scorer,
            #     logits_processor=logits_processor,
            #     stopping_criteria=stopping_criteria,
            #     pad_token_id=generation_config.pad_token_id,
            #     eos_token_id=generation_config.eos_token_id,
            #     output_scores=generation_config.output_scores,
            #     return_dict_in_generate=generation_config.return_dict_in_generate,
            #     synced_gpus=synced_gpus,
            #     **model_kwargs,
            # )

        elif is_constraint_gen_mode:
            raise NotImplementedError("Constrained generation is not supported with perturbations.")
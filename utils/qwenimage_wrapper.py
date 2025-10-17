import os
import math
import types
import json
from typing import List, Optional, Tuple
import numpy as np
import torch
from torch import nn

from utils.scheduler import SchedulerInterface, FlowMatchScheduler

from diffusers import QwenImagePipeline, QwenImageTransformer2DModel, FlowMatchEulerDiscreteScheduler
from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration
from safetensors.torch import load_file


class QwenImageTextEncoder(nn.Module):
    def __init__(self, model_name="Qwen-Image") -> None:
        super().__init__()
        self.model_name = model_name

        self.text_encoder: Qwen2_5_VLForConditionalGeneration = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            subfolder='text_encoder',
            dtype=torch.bfloat16,
        )
        self.text_encoder.eval()

        self.tokenizer: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(
            model_name,
            subfolder='tokenizer',
        )
        self.pipeline: QwenImagePipeline = QwenImagePipeline.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            scheduler=None,
            vae=None,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=None,
        )

    @property
    def device(self):
        return next(self.text_encoder.parameters()).device

    def forward(self, text_prompts: List[str]) -> dict:
        prompt_embeds, prompt_embeds_mask = self.pipeline.encode_prompt(
            prompt=text_prompts,
            prompt_embeds=None,
            prompt_embeds_mask=None,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )

        return {
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
        }


class QwenImageWrapper(nn.Module):
    def __init__(
            self,
            model_name="Qwen-Image",
            timestep_shift=8.0,
            num_inference_steps=4,
            pretrain_weight=None,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.model_name = model_name

        with open(os.path.join(model_name, 'transformer', 'config.json')) as f:
            kw = json.load(f)
        kw.pop('_class_name')
        kw.pop('_diffusers_version')
        kw.pop('pooled_projection_dim')
        # kw['num_layers'] = 1

        self.model = QwenImageTransformer2DModel(**kw)
        if pretrain_weight is not None:
            sd = load_file(pretrain_weight)
            self.model.load_state_dict(sd, strict=True)

        # self.scheduler = FlowMatchScheduler(
        #     shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        # )
        # self.scheduler.set_timesteps(1000, training=True)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            # {
            # "base_image_seq_len": 256,
            # "base_shift": math.log(3),  # We use shift=3 in distillation
            # "invert_sigmas": False,
            # "max_image_seq_len": 8192,
            # "max_shift": math.log(3),  # We use shift=3 in distillation
            # "num_train_timesteps": 1000,
            # "shift": timestep_shift,
            # "shift_terminal": None,  # set shift_terminal to None
            # "stochastic_sampling": False,
            # "time_shift_type": "exponential",
            # "use_beta_sigmas": False,
            # "use_dynamic_shifting": False,
            # "use_exponential_sigmas": False,
            # "use_karras_sigmas": False,
            # }
            {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
            }
        )

        # sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        # timesteps, num_inference_steps = retrieve_timesteps(
        #     self.scheduler,
        #     num_inference_steps,
        #     sigmas=sigmas,
        #     mu=math.log(3),
        # )
        self.timesteps = self.scheduler.timesteps
        self.sigmas = self.scheduler.sigmas

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor,
                                 timestep: torch.Tensor) -> torch.Tensor:
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        original_device = flow_pred.device

        flow_pred = flow_pred.to(dtype=torch.float32, device=original_device)
        xt = xt.to(dtype=torch.float32, device=original_device)
        sigmas = self.sigmas.to(dtype=torch.float32, device=original_device)
        timesteps = self.timesteps.to(dtype=torch.float32, device=original_device)

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(),
            dim=1,
        )
        sigma_t = sigmas[timestep_id]
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor,
                                 timestep: torch.Tensor) -> torch.Tensor:
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        original_device = x0_pred.device

        x0_pred = x0_pred.to(dtype=torch.float32, device=original_device)
        xt = xt.to(dtype=torch.float32, device=original_device)
        sigmas = sigmas.to(dtype=torch.float32, device=original_device)
        timesteps = timesteps.to(dtype=torch.float32, device=original_device)

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(),
            dim=1,
        )
        sigma_t = sigmas[timestep_id]
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)


    def forward(
            self,
            noisy_image_or_video: torch.Tensor,  # [b, img_s, 64]
            conditional_dict: dict,  # [b, txt_s, 3584], [b, txt_s, 3584]
            timestep: torch.Tensor,  # [b]
            img_shapes: List[Tuple[int, int, int]],  # [1, img_h//16, img_w//16]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt_embeds = conditional_dict["prompt_embeds"]
        prompt_embeds_mask = conditional_dict["prompt_embeds_mask"]
        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

        # [B]
        input_timestep = timestep.float() / 1000

        # X0 prediction
        # [b, img_s, 64]
        flow_pred = self.model(
            hidden_states=noisy_image_or_video,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask,
            timestep=input_timestep,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            guidance=None,
            attention_kwargs=None,
            controlnet_block_samples=None,
            return_dict=False,
        )[0]

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred,
            xt=noisy_image_or_video,
            timestep=timestep
        )

        return flow_pred, pred_x0


QwenImageScheduler = FlowMatchEulerDiscreteScheduler
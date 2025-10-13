import os
import types
import json
from typing import List, Optional, Tuple
import torch
from torch import nn

from utils.scheduler import SchedulerInterface, FlowMatchScheduler

from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration


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
            *args,
            **kwargs,
    ):
        super().__init__()
        self.model_name = model_name

        with open(os.path.join(model_name, 'transformer', 'config.json')) as f:
            kw = json.load(f)
        kw.pop('_class_name')
        kw.pop('_diffusers_version')

        self.model = QwenImageTransformer2DModel(**kw)
        self.model.eval()

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor,
                                 timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor,
                                 timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device),
            [x0_pred, xt, scheduler.sigmas, scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1)
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
        input_timestep = timestep

        # X0 prediction
        # [b, img_s, 64]
        flow_pred = self.model(
            hidden_states=noisy_image_or_video,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_embeds_mask,
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

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()

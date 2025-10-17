from typing import List, Tuple
import torch
from torch import int16, nn
from utils.wan_wrapper import WanDiffusionWrapper
from utils.qwenimage_wrapper import QwenImageWrapper
from utils.scheduler import SchedulerInterface
import torch.distributed as dist


class BidirectionalTrainingPipeline(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        denoising_step_list: List[int],
        scheduler: SchedulerInterface,
        generator: WanDiffusionWrapper,
    ):
        super().__init__()
        self.model_name = model_name
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]

    def generate_and_sync_list(self, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(1,),
                device=device
            )
        else:
            indices = torch.empty(1, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()

    def inference_with_trajectory(self, noise: torch.Tensor, clip_fea, y, **conditional_dict) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """

        # initial point
        noisy_image_or_video = noise
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(num_denoising_steps, device=noise.device)

        # use the last n-1 timesteps to simulate the generator's input
        for index, current_timestep in enumerate(self.denoising_step_list):
            exit_flag = (index == exit_flags[0])
            timestep = torch.ones(
                noise.shape[:2],
                device=noise.device,
                dtype=torch.int64) * current_timestep
            if not exit_flag:
                with torch.no_grad():
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_image_or_video,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        clip_fea=clip_fea,
                        y=y
                    )  # [B, F, C, H, W]

                    next_timestep = self.denoising_step_list[index + 1] * torch.ones(
                        noise.shape[:2], dtype=torch.long, device=noise.device)

                    noisy_image_or_video = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep.flatten(0, 1)
                    ).unflatten(0, denoised_pred.shape[:2])
            else:
                _, denoised_pred = self.generator(
                    noisy_image_or_video=noisy_image_or_video,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    clip_fea=clip_fea,
                    y=y
                )  # [B, F, C, H, W]
                break

        if exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()

        return denoised_pred, denoised_timestep_from, denoised_timestep_to


class BidirectionalTrainingT2IPipeline(nn.Module):
    def __init__(
            self,
            model_name: str,
            denoising_step_list: List[int],
            scheduler: SchedulerInterface,
            generator: QwenImageWrapper,
    ):
        super().__init__()
        self.model_name = model_name
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]

    def generate_and_sync_list(self, num_denoising_steps: int, device: torch.device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(1,),
                dtype=torch.long,
                device=device,
            )
        else:
            indices = torch.empty(
                (1, ),
                dtype=torch.long,
                device=device,
            )

        dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()

    def inference_with_trajectory(
            self,
            noise: torch.Tensor,
            img_shapes: List[List[Tuple[int, int, int]]],  # [[1, img_h//16, img_w//16]]
            **conditional_dict,
    ) -> torch.Tensor:
        # initial point
        noisy_image_or_video = noise
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(num_denoising_steps, device=noise.device)

        # use the last n-1 timesteps to simulate the generator's input
        for index, current_timestep in enumerate(self.denoising_step_list):
            exit_flag = (index == exit_flags[0])
            timestep = torch.full(
                (noise.size(0),),
                fill_value=current_timestep,
                dtype=torch.long,
                device=noise.device,
            )
            if noise.device.index == 0:
                print(f'train generator: {timestep=}\n', end='')
            if not exit_flag:
                with torch.no_grad():
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_image_or_video,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        img_shapes=img_shapes,
                    )  # [B, img_s, 64]

                    next_timestep = torch.full(
                        (noise.size(0),),
                        fill_value=self.denoising_step_list[index + 1],
                        dtype=torch.long,
                        device=noise.device,
                    )
                    noisy_image_or_video = self.scheduler.add_noise(
                        denoised_pred,
                        torch.randn_like(denoised_pred),
                        next_timestep,
                    )
            else:
                _, denoised_pred = self.generator(
                    noisy_image_or_video=noisy_image_or_video,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    img_shapes=img_shapes,
                )  # [B, img_s, 64]
                break

        if exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (
                        self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()
                ).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (
                        self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()
                ).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (
                        self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()
                ).abs(), dim=0).item()

        return denoised_pred, denoised_timestep_from, denoised_timestep_to

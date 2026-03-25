"""
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import einops
import torch
import os
from typing import Any, Dict, List, Tuple, Union
import numpy as np

from einops import rearrange
from sgm.models.diffusion import DiffusionEngine
from sgm.util import instantiate_from_config
from safetensors.torch import load_file as load_safetensors
from sgm.util import disabled_train
from torch.optim.lr_scheduler import LambdaLR

from m2svid.utils.anaglyph import make_anaglyph_video
from m2svid.metrics import psnr
from pytorch_msssim import ms_ssim

import random
from sgm.modules.autoencoding.lpips.loss.lpips import LPIPS


def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


class VideoLDM(DiffusionEngine):
    def __init__(
            self,
            num_samples,
            trained_param_keys=[''],
            devide_loss_on_accumulate_grad_batches=1,
            ucg_keys=None,

            cond_reprojected_video=False,
            cond_video_2nd_view=False,

            clip_condition='video_2nd_view',
            clip_condition_all_frames=False,

            cond_randomly_gt_frames_in_reprojected_video=False,
            gt_video_sampling_strategy='random_frames',
            gt_video_sampling_exp_scale=1.0,

            apply_loss_on_images = False,
            *args,
            **kwargs
        ):
        self.trained_param_keys = trained_param_keys
        self.apply_loss_on_images = apply_loss_on_images

        super().__init__(*args, **kwargs)
        self.num_samples = num_samples
        self.devide_loss_on_accumulate_grad_batches = devide_loss_on_accumulate_grad_batches
        self.cond_reprojected_video = cond_reprojected_video
        self.cond_video_2nd_view = cond_video_2nd_view
        self.clip_condition_all_frames = clip_condition_all_frames
        self.clip_condition = clip_condition
        self.ucg_keys = ucg_keys
        self.gt_video_sampling_strategy = gt_video_sampling_strategy
        self.gt_video_sampling_exp_scale = gt_video_sampling_exp_scale
        self.cond_randomly_gt_frames_in_reprojected_video = cond_randomly_gt_frames_in_reprojected_video

        self.additional_metric = {
            'lpips':  LPIPS().eval()
        }

    def init_from_ckpt(
        self,
        path: str,
    ) -> None:
        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")
            if "state_dict" in sd:
                sd = sd["state_dict"]
        elif path.endswith("pt"):
            sd_raw = torch.load(path, map_location="cpu")
            sd = {}
            for k in sd_raw['module']:
                sd[k[len('module.'):]] = sd_raw['module'][k]
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            raise NotImplementedError

        # missing, unexpected = self.load_state_dict(sd, strict=True)
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                params = params + list(embedder.parameters())
        opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt


    @torch.no_grad()
    def add_custom_cond(self, batch, infer=False):
        _, _, num_samples, _, _ = batch['video'].shape
        batch['num_video_frames'] = num_samples
        if num_samples > self.num_samples:
            raise NotImplementedError

        if self.clip_condition_all_frames:
            batch['cond_frames_without_noise'] = rearrange(batch[self.clip_condition], "b c t h w -> (b t) c h w", t=num_samples).half()
        else:
            image = batch[self.clip_condition][:, :, 0]
            batch['cond_frames_without_noise'] = image.half()

        N = batch['video'].shape[0]
        if not infer:
            cond_aug = ((-3.0) + (0.5) * torch.randn((N,))).exp().cuda().half()
        else:
            cond_aug = torch.full((N, ), 0.02).cuda().half()
        batch['cond_aug'] = cond_aug

        def preprocess_cond(cond_aug, video):
            video = (video + rearrange(cond_aug, 'b -> b 1 1 1 1') * torch.randn_like(video)).half()
            video = rearrange(video, "b c t h w -> (b t) c h w", t=num_samples)
            return video

        if self.cond_reprojected_video:
            batch['cond_reprojected_video'] = preprocess_cond(cond_aug, batch['reprojected_video'])
            inpainting_mask = batch["reprojected_mask"]
            inpainting_mask = einops.rearrange(inpainting_mask, 'b 1 t h w -> (b t) h w')
            batch['inpainting_mask'] = inpainting_mask
            # batch['original_reprojected_mask']  = rearrange(batch['original_reprojected_mask'], 'b c t h w -> (b t) c h w')[:, 0]


            if (not infer) and self.cond_randomly_gt_frames_in_reprojected_video:
                if isinstance(self.gt_video_sampling_strategy, str):
                    gt_video_sampling_strategy = self.gt_video_sampling_strategy
                else:
                    gt_video_sampling_strategy = np.random.choice(self.gt_video_sampling_strategy)

                frame_ids = []
                if random.random() < 0.5:
                    if gt_video_sampling_strategy == 'first_3frames':
                        for i in range(N):
                            frame_ids.extend(list(range(i * num_samples, i * num_samples + 3)))
                    elif gt_video_sampling_strategy == 'random_frames':
                        n_gt_frames = N * int(np.round(np.random.exponential(scale=self.gt_video_sampling_exp_scale)))
                        n_gt_frames = min(n_gt_frames, N * num_samples)
                        if n_gt_frames > 0:
                            frame_ids = np.random.choice(list(range(N * num_samples)), n_gt_frames)
                    elif gt_video_sampling_strategy == 'random_first_frames':
                        for i in range(N):
                            n_gt_frames = int(np.round(np.random.exponential(scale=self.gt_video_sampling_exp_scale)))
                            n_gt_frames = min(n_gt_frames, num_samples)
                            frame_ids.extend(list(range(i * num_samples, i * num_samples + n_gt_frames)))
                    else:
                        raise NotImplementedError
                original_video = rearrange(batch['video'], "b c t h w -> (b t) c h w", t=num_samples)
                for frame_id in frame_ids:
                    batch['cond_reprojected_video'][frame_id] = original_video[frame_id]
                    batch['inpainting_mask'][frame_id] = -1

        if self.cond_video_2nd_view:
            batch['cond_video_2nd_view'] = preprocess_cond(cond_aug, batch['video_2nd_view'])

        # for dataset without indicator
        if not 'image_only_indicator' in batch:
            batch['image_only_indicator'] = torch.zeros((N, num_samples)).cuda().half()
        return batch

    def forward(self, x, batch):
        if self.apply_loss_on_images:
            loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch,
                                first_stage_model=self.first_stage_model,
                                scale_factor=self.scale_factor,
                                apply_first_stage=self.apply_loss_on_images,
                                disable_first_stage_autocast=self.disable_first_stage_autocast,
                                en_and_decode_n_samples_a_time=self.en_and_decode_n_samples_a_time)
        else:
            loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)

        if isinstance(loss, tuple) or isinstance(loss, list):
            assert len(loss) == 2
            loss, loss_dict = loss
        else:
            loss_dict = {}

        loss_mean = loss.mean()
        loss_grad_acc_fixed = loss_mean / self.devide_loss_on_accumulate_grad_batches

        loss_dict["loss"] = loss_mean
        loss_dict["loss_grad_acc_fixed"] = loss_grad_acc_fixed

        return loss_grad_acc_fixed, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch) # b c t h w
        batch = self.add_custom_cond(batch)

        x = rearrange(x, 'b c t h w -> (b t) c h w')
        if not self.apply_loss_on_images:
            x = self.encode_first_stage(x)

        batch["global_step"] = self.global_step
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            loss, loss_dict = self(x, batch)
        return loss, loss_dict

    @torch.no_grad()
    def generate(
        self,
        batch: Dict,
        ucg_keys: List[str] = None,
        do_not_decode = False,
        **kwargs,
    ) -> Dict:
        device = torch.device('cuda')
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys

        _, _, t_frames, _, _ = batch['video'].shape
        if t_frames > self.num_samples:
            raise NotImplementedError

        batch = self.add_custom_cond(batch, infer=True)
        frames = self.get_input(batch)
        N = len(frames)

        # CPU offload: move conditioner to GPU
        self.conditioner.to(device)

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys if len(self.conditioner.embedders) > 0 else [],
        )

        # CPU offload: move conditioner back to CPU
        self.conditioner.to('cpu')
        torch.cuda.empty_cache()

        x = rearrange(frames, 'b c t h w -> (b t) c h w')
        x = x.to(device)

        # if not do_not_decode:
        #     z = self.encode_first_stage(x.half())
        #     x_rec = self.decode_first_stage(z.half(), num_video_frames=batch["num_video_frames"])
        #     x_rec = rearrange(x_rec, '(b t) c h w -> b c t h w', t=batch["num_video_frames"])
        # else:
        #     x_rec = None

        additional_model_inputs = {}
        additional_model_inputs["image_only_indicator"] = torch.zeros(N * 2, batch["num_video_frames"]).to(device)
        additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

        if self.cond_reprojected_video:
            inpainting_mask = batch["inpainting_mask"]
            inpainting_mask = torch.concat([inpainting_mask, inpainting_mask], dim=0)
            additional_model_inputs["inpainting_mask"] = inpainting_mask

        def denoiser(input, sigma, c):
            return self.denoiser(self.model, input, sigma, c, **additional_model_inputs)

        # CPU offload: move model to GPU
        if hasattr(self, 'model'): self.model.to(device)

        with self.ema_scope("Plotting"):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                shape = (x.shape[0], 4, int(x.shape[2] // 8), int(x.shape[3] // 8))
                randn = torch.randn(shape, device=device)
                samples = self.sampler(denoiser, randn, cond=c, uc=uc, num_video_frames=batch["num_video_frames"])

        # CPU offload: move model back to CPU
        if hasattr(self, 'model'): self.model.to('cpu')
        torch.cuda.empty_cache()

        if not do_not_decode:
            # CPU offload: move first stage to GPU
            if hasattr(self, 'first_stage_model'): self.first_stage_model.to(device)
            samples = self.decode_first_stage(samples.half(), num_video_frames=batch["num_video_frames"])
            # CPU offload: move first stage back to CPU
            if hasattr(self, 'first_stage_model'): self.first_stage_model.to('cpu')
            torch.cuda.empty_cache()
            samples = einops.rearrange(samples, '(b t) c h w -> b c t h w', t=batch["num_video_frames"])

        output = {
            'gt-video': frames,
            # 'reconstructed-gt-video': x_rec,
            'generated-video': samples,
            'c': c,
            'uc': uc
        }

        return output


    def test_step(self, batch, batch_idx):
        output = self.generate(batch, self.ucg_keys)

        gt_video = output['gt-video']
        generated_video = output['generated-video']

        inpainting_mask_for_testing = batch['inpainting_mask_for_testing']
        inpainting_mask_for_testing = (inpainting_mask_for_testing > 0).to(inpainting_mask_for_testing.dtype)
        gt_video_inpainted = gt_video * inpainting_mask_for_testing + (1. - inpainting_mask_for_testing)
        generated_video_inpainted = generated_video * inpainting_mask_for_testing + (1. - inpainting_mask_for_testing)

        gt_video_reprojected = gt_video * (1. - inpainting_mask_for_testing) + inpainting_mask_for_testing
        generated_video_reprojected = generated_video * (1. - inpainting_mask_for_testing) + inpainting_mask_for_testing

        metrics = {}
        metrics['total'] = torch.tensor([len(gt_video)]).to(generated_video.device)
        metrics['total_batches'] = torch.tensor([1]).to(generated_video.device)

        h, w = generated_video.shape[-2], generated_video.shape[-1]

        triplets = [
            (gt_video, generated_video, None, ''),
            (gt_video_inpainted, generated_video_inpainted, None, '_inpainted_only'),
            (gt_video_reprojected, generated_video_reprojected, None, '_reprojected_only'),
        ]

        lpips = self.additional_metric['lpips'].to('cuda')

        for ref_video, gen_video, mask, suffix in triplets:
            ref_video = ((ref_video + 1.0) / 2.0).clamp(0, 1)
            gen_video = ((gen_video + 1.0) / 2.0).clamp(0, 1)
            metrics.update({
                f'test/psnr{suffix}': psnr(ref_video, gen_video, data_range=1.0),
                f'test/ms_ssim{suffix}': ms_ssim(ref_video, gen_video, data_range=1.0),
                f'test/lpips{suffix}': lpips(
                    einops.rearrange(ref_video.float() * 2.0 - 1.0, 'b c t h w -> (b t) c h w'),
                    einops.rearrange(gen_video.float()  * 2.0 - 1.0, 'b c t h w -> (b t) c h w'),
                ).mean(),
            })

        for key, val in metrics.items():
            self.log(key, val,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=False)
        return metrics

    def test_epoch_end(self, outputs):
        results = {}
        for key in list(outputs[0].keys()):
            value = torch.stack([x[key] for x in outputs]).sum()
            results[key] = value

        for key in list(outputs[0].keys()):
            if key not in ['total', 'total_batches']:
                results[key] = results[key] / results['total_batches']

        for key, val in results.items():
            self.log(key + '_avg', val,
                    prog_bar=True,
                    logger=True,
                    on_step=False,
                    on_epoch=True)

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        return self.test_epoch_end(outputs)

    @torch.no_grad()
    def log_images(
        self,
        batch: Dict,
        ucg_keys: List[str] = None,
        **kwargs,
    ) -> Dict:
        output = self.generate(batch, ucg_keys)
        c = output.pop('c')
        uc = output.pop('uc')

        videos = []
        if self.cond_video_2nd_view:
            video = rearrange(batch['cond_video_2nd_view'], '(b t) c h w -> b c t h w', t=batch['num_video_frames'])
            videos.append(video)
        if self.cond_reprojected_video:
            video = rearrange(batch['cond_reprojected_video'], '(b t) c h w -> b c t h w', t=batch['num_video_frames'])
            videos.append(video)
        videos.append(output["generated-video"])
        # videos.append(output["reconstructed-gt-video"])
        output["combined-video"] = torch.cat(videos, dim=4)

        sbs_videos = [batch['video_2nd_view'], output["generated-video"]]
        output["sbs"] = torch.cat(sbs_videos, dim=4)

        def make_anaglyph_video_batch(left, right):
            return torch.stack([make_anaglyph_video(left[i], right[i], unnormalized_videos=True) for i in range(len(left))])

        # videos[-2] = make_anaglyph_video_batch(batch['video_2nd_view'], videos[-2])
        videos[-1] = make_anaglyph_video_batch(batch['video_2nd_view'], videos[-1])

        output["combined-anaglyph-video"] = torch.cat(videos, dim=4)
        return output

    def configure_optimizers(self):
        lr = self.learning_rate
        if 'all' in self.trained_param_keys:
            params = list(self.model.parameters())
        else:
            names = []
            params = []
            for name, param in self.model.named_parameters():
                flag = False
                for k in self.trained_param_keys:
                    if k in name:
                        names += [name]
                        params += [param]
                        flag = True
                    if flag:
                        break
            print("Trainable diffuion model params: ", names)

        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                params = params + list(embedder.parameters())

        opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

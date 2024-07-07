from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torchvision.transforms as TT
import numpy as np
import argparse
import os
import time
from tqdm import tqdm


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def get_views(height, width, window_size, stride):
    num_blocks_height = (height - window_size) // stride + 1
    num_blocks_width = (width - window_size) // stride + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size
        views.append((h_start, h_end, w_start, w_end))
    return views


class TwinDiffusion(nn.Module):
    def __init__(self, device, sd_version='2.0', hf_key=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        if hf_key is not None:
            print(f'Using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        elif self.sd_version == 'xl-1.0':
            model_key = "stabilityai/stable-diffusion-xl-base-1.0"
        else:
            raise ValueError(f'Stable Diffusion Version {self.sd_version} NOT Supported.')

        print('Loading stable diffusion...')

        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        if 'xl' in self.sd_version:
            self.tokenizer_2 = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer_2")
            self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_key, subfolder="text_encoder_2").to(self.device)

        print('Loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        if 'xl' in self.sd_version:
            tokenizers = [self.tokenizer, self.tokenizer_2]
            text_encoders = [self.text_encoder, self.text_encoder_2]
            text_embeddings_list = []
            uncond_text_embeddings_list = []
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                text_input = tokenizer(prompts, padding='max_length', max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')
                text_embeddings = text_encoder(text_input.input_ids.to(self.device), output_hidden_states=True)
                pooled_text_embeddings = text_embeddings[0]
                text_embeddings = text_embeddings.hidden_states[-2]
                text_embeddings_list.append(text_embeddings)

                uncond_input = tokenizer(negative_prompts, padding='max_length', max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')
                uncond_embeddings = text_encoder(uncond_input.input_ids.to(self.device), output_hidden_states=True)
                pooled_uncond_embeddings = uncond_embeddings[0]
                uncond_embeddings = uncond_embeddings.hidden_states[-2]
                uncond_text_embeddings_list.append(uncond_embeddings)

            text_embeddings = torch.cat(text_embeddings_list, dim=-1)
            uncond_embeddings = torch.cat(uncond_text_embeddings_list, dim=-1)

            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            add_text_embeds = torch.cat([pooled_uncond_embeddings, pooled_text_embeddings])
            return text_embeddings, add_text_embeds
        else:
            text_input = self.tokenizer(prompts, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

            uncond_input = self.tokenizer(negative_prompts, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            return text_embeddings

    @torch.no_grad()
    def decode_latents(self, latents):
        imgs = self.vae.decode(1 / self.vae.config.scaling_factor * latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        imgs = imgs.cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).astype(np.uint8)
        return imgs

    @torch.no_grad()
    def denoise_single_step(self, latents, t, text_embeds, guidance_scale, added_cond_kwargs=None):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)
        if 'xl' in self.sd_version:
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds, added_cond_kwargs=added_cond_kwargs).sample
        else:
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds).sample
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        return latents

    def text2panorama_optm(self, prompts, negative_prompts, height=512, width=2048, lam=1.0, view_stride=16, cross_time=2, num_inference_steps=50):
        """
        height, width：the size of panoramas
        lam：the Lagrange multiplier of Crop Fusion function
        view_stride：the step size when cropping panoramas
        cross_time：the frequency of Cross Sampling
        """
        guidance_scale = 5.0 if 'xl' in self.sd_version else 7.5
        window_size = 128 if 'xl' in self.sd_version else 64

        prompts = [prompts] if isinstance(prompts, str) else prompts
        negative_prompts = [negative_prompts] if isinstance(negative_prompts, str) else negative_prompts

        batch_size = len(prompts)

        if 'xl' in self.sd_version:
            text_embeds, add_text_embeds = self.get_text_embeds(prompts, negative_prompts)

            add_time_ids = torch.tensor([[height, width, 0, 0, height, width]], dtype=text_embeds.dtype, device=self.device)
            negative_add_time_ids = add_time_ids
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids])
            add_time_ids = add_time_ids.repeat(batch_size, 1)

            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        else:
            text_embeds = self.get_text_embeds(prompts, negative_prompts)

        self.scheduler.set_timesteps(num_inference_steps)

        latents = torch.randn((batch_size, self.unet.config.in_channels, height // 8, width // 8), device=self.device)
        latents = latents * self.scheduler.init_noise_sigma

        views = get_views(height // 8, width // 8, window_size=window_size, stride=view_stride)
        count = torch.zeros_like(latents)
        value = torch.zeros_like(latents)

        # Cross Sampling
        cross_stride = 0
        all_views = [views]
        all_cross_strides = [cross_stride]
        for _ in range(cross_time - 1):
            cross_stride += view_stride // cross_time
            views_cross = [views[0]] + [(hs, he, ws + cross_stride, we + cross_stride) for hs, he, ws, we in views[1:-1]] + [views[-1]]
            all_views.append(views_cross)
            all_cross_strides.append(cross_stride)

        with tqdm(self.scheduler.timesteps, desc='Generating images') as pbar:
            for i, t in enumerate(pbar):
                count.zero_(), value.zero_()

                for idx, view in enumerate(all_views[i % cross_time]):
                    h_start, h_end, w_start, w_end = view
                    latents_view = latents[:, :, h_start:h_end, w_start:w_end]

                    if 'xl' in self.sd_version:
                        latents_view = self.denoise_single_step(latents_view, t, text_embeds, guidance_scale, added_cond_kwargs=added_cond_kwargs)
                    else:
                        latents_view = self.denoise_single_step(latents_view, t, text_embeds, guidance_scale)

                    # Crop Fusion
                    if idx > 0 and i < num_inference_steps // 2:
                        latents_view_pre = latents_view.clone().detach()

                        # training-based optimization
                        # latents_view.requires_grad = True
                        # optimizer = torch.optim.Adam([latents_view], lr=1e-5 * (1. - i / 100.))
                        #
                        # for epoch in range(train_epochs):
                        #     if idx == 1:
                        #         loss = nnf.mse_loss(nbr_views_optm[:, :, :, view_stride + all_cross_strides[i % cross_time]:], latents_view[:, :, :, :window_size - view_stride - all_cross_strides[i % cross_time]]) + \
                        #                lam * nnf.mse_loss(latents_view_pre, latents_view)
                        #     elif idx == len(views) - 1:
                        #         loss = nnf.mse_loss(nbr_views_optm[:, :, :, view_stride - all_cross_strides[i % cross_time]:], latents_view[:, :, :, :window_size - view_stride + all_cross_strides[i % cross_time]]) + \
                        #                lam * nnf.mse_loss(latents_view_pre, latents_view)
                        #     else:
                        #         loss = nnf.mse_loss(nbr_views_optm[:, :, :, view_stride:], latents_view[:, :, :, :window_size - view_stride]) + \
                        #                lam * nnf.mse_loss(latents_view_pre, latents_view)
                        #
                        #     optimizer.zero_grad()
                        #     loss.backward()
                        #     optimizer.step()

                        # training-free optimization
                        if idx == 1:
                            latents_view[:, :, :, :window_size - view_stride - all_cross_strides[i % cross_time]] = (nbr_views_optm[:, :, :, view_stride + all_cross_strides[i % cross_time]:] +
                                                                                                                     lam * latents_view_pre[:, :, :, :window_size - view_stride - all_cross_strides[i % cross_time]]) / (1 + lam)
                        elif idx == len(views) - 1:
                            latents_view[:, :, :, :window_size - view_stride + all_cross_strides[i % cross_time]] = (nbr_views_optm[:, :, :, view_stride - all_cross_strides[i % cross_time]:] +
                                                                                                                     lam * latents_view_pre[:, :, :, :window_size - view_stride + all_cross_strides[i % cross_time]]) / (1 + lam)
                        else:
                            latents_view[:, :, :, :window_size - view_stride] = (nbr_views_optm[:, :, :, view_stride:] +
                                                                                 lam * latents_view_pre[:, :, :, :window_size - view_stride]) / (1 + lam)

                    value[:, :, h_start:h_end, w_start:w_end] += latents_view
                    count[:, :, h_start:h_end, w_start:w_end] += 1

                    nbr_views_optm = latents_view.clone()  # reserving the left neighbor z^{i-1} for z^i

                latents = torch.where(count > 0, value / count, value)

        imgs = self.decode_latents(latents)
        return imgs


if __name__ == '__main__':
    # sd：
    # H, W = 512, 2048
    # view_stride = 16
    # sdxl：
    # H, W = 1024, 4096
    # view_stride = 32

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default="A photo of the dolomites")
    parser.add_argument('--negative', type=str, default="")
    parser.add_argument('--sd_version', type=str, default='2.0', choices=['1.5', '2.0', '2.1', 'xl-1.0'])
    parser.add_argument('--H', type=int, default=512)
    parser.add_argument('--W', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--lam', type=float, default=1.0)
    parser.add_argument('--view_stride', type=int, default=16)
    parser.add_argument('--cross_time', type=int, default=2)
    parser.add_argument('--n', type=int, default=1)
    opt = parser.parse_args()

    if opt.seed != -1:
        seed_everything(opt.seed)

    device = torch.device('cuda')
    td = TwinDiffusion(device, opt.sd_version)

    start = time.time()
    imgs = td.text2panorama_optm([opt.prompt] * opt.n, [opt.negative] * opt.n, opt.H, opt.W, lam=opt.lam, view_stride=opt.view_stride)
    print(f"time: {time.time() - start} s")

    for i in tqdm(range(opt.n), desc='Saving images'):
        TT.ToPILImage()(imgs[i]).save(f"out{i}.png")


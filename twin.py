from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
import torch
import torch.nn as nn
import torchvision.transforms as TT
import torch.nn.functional as nnf
import numpy as np
import argparse
import os
from tqdm import tqdm
from utils import view_images


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


class StableDiffusion(nn.Module):
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

    @torch.no_grad()
    def decode_latents(self, latents):
        imgs = self.vae.decode(1 / self.vae.config.scaling_factor * latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        imgs = imgs.cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).astype(np.uint8)
        return imgs

    def generate_twin_images(self, prompts, negative_prompts, lam=1.0, num_inference_steps=50):
        height = width = 1024 if 'xl' in self.sd_version else 512
        guidance_scale = 5.0 if 'xl' in self.sd_version else 7.5

        prompts = [prompts] if isinstance(prompts, str) else prompts
        negative_prompts = [negative_prompts] if isinstance(negative_prompts, str) else negative_prompts

        batch_size = len(prompts)

        if 'xl' in self.sd_version:
            text_embeds_1_2, add_text_embeds_1_2 = self.get_text_embeds(prompts * 2, negative_prompts * 2)
            text_embeds, add_text_embeds = self.get_text_embeds(prompts, negative_prompts)

            add_time_ids = torch.tensor([[height, width, 0, 0, height, width]], dtype=text_embeds.dtype, device=self.device)
            negative_add_time_ids = add_time_ids
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids])
            add_time_ids_1_2 = add_time_ids.repeat(2 * batch_size, 1)
            add_time_ids = add_time_ids.repeat(batch_size, 1)

            added_cond_kwargs_1_2 = {"text_embeds": add_text_embeds_1_2, "time_ids": add_time_ids_1_2}
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        else:
            text_embeds_1_2 = self.get_text_embeds(prompts * 2, negative_prompts * 2)
            text_embeds = self.get_text_embeds(prompts, negative_prompts)

        self.scheduler.set_timesteps(num_inference_steps)

        latent_1_2 = torch.randn((batch_size, 2, self.unet.config.in_channels, height // 8, width // 8), device=self.device)  # z1 z2
        latent_1_2 = latent_1_2 * self.scheduler.init_noise_sigma
        latent_1_2[:, 1, :, :, :width // 16] = latent_1_2[:, 0, :, :, width // 16:]  # initialize [z2]l=[z1]r
        latent_2_optm = latent_1_2[:, 1].clone()  # initialize z2*=z2
        latent_1_2 = latent_1_2.reshape(2 * batch_size, *latent_1_2.shape[2:])

        with tqdm(self.scheduler.timesteps, desc='generating images') as pbar:
            for i, t in enumerate(pbar):
                if 'xl' in self.sd_version:
                    latent_1_2 = self.denoise_single_step(latent_1_2, t, text_embeds_1_2, guidance_scale, added_cond_kwargs_1_2)
                    latent_2_optm = self.denoise_single_step(latent_2_optm, t, text_embeds, guidance_scale, added_cond_kwargs)
                else:
                    latent_1_2 = self.denoise_single_step(latent_1_2, t, text_embeds_1_2, guidance_scale)
                    latent_2_optm = self.denoise_single_step(latent_2_optm, t, text_embeds, guidance_scale)

                if i < num_inference_steps // 2:  # see Ablation.1
                    latent_2_optm_pre = latent_2_optm.clone().detach()  # z2* that hasn't been optimized yet
                    latent_1_2 = latent_1_2.reshape(batch_size, 2, *latent_1_2.shape[1:])

                    # training-based optimization
                    # latent_2_optm.requires_grad = True
                    # optimizer = torch.optim.Adam([latent_2_optm], lr=1e-3 * (1. - i / 100.))
                    #
                    # for epoch in range(train_epochs):
                    #     loss = nnf.mse_loss(latent_1_2[:, 0, :, :, width // 16:], latent_2_optm[:, :, :, :width // 16]) + \
                    #            lam * nnf.mse_loss(latent_2_optm_pre, latent_2_optm)
                    #     # limit test of the function
                    #     # loss = nnf.mse_loss(latent_1_2[:, 0, :, :, width // 16:], latent_2_optm[:, :, :, :width // 16]) + \
                    #     #        lam * nnf.mse_loss(latent_1_2[:, 1], latent_2_optm)
                    #
                    #     pbar.set_postfix({'epoch': epoch, 'loss': loss.item() / batch_size})
                    #
                    #     optimizer.zero_grad()
                    #     loss.backward()
                    #     optimizer.step()

                    # training-free optimization
                    latent_2_optm[:, :, :, :width // 16] = (latent_1_2[:, 0, :, :, width // 16:] +
                                                            lam * latent_2_optm_pre[:, :, :, :width // 16]) / (1 + lam)
                    # limit test
                    # latent_2_optm[:, :, :, :width // 16] = (latent_1_2[:, 0, :, :, width // 16:] +
                    #                                         lam * latent_1_2[:, 1, :, :, :width // 16]) / (1 + lam)

                    latent_1_2 = latent_1_2.reshape(2 * batch_size, *latent_1_2.shape[2:])

        latents = torch.cat([latent_1_2.reshape(batch_size, 2, *latent_1_2.shape[1:]), latent_2_optm.unsqueeze(1)], dim=1)
        imgs = self.decode_latents(latents.reshape(3 * batch_size, *latents.shape[2:]))
        imgs = imgs.reshape(batch_size, 3, *imgs.shape[1:])  # return I1、I2、I2*,  I1 and I2* are twin images
        return imgs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default="A photo of the dolomites")
    parser.add_argument('--negative', type=str, default="")
    parser.add_argument('--sd_version', type=str, default='2.0', choices=['1.5', '2.0', '2.1', 'xl-1.0'])
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--lam', type=float, default=1.0)
    parser.add_argument('--n', type=int, default=3)
    opt = parser.parse_args()

    if opt.seed != -1:
        seed_everything(opt.seed)

    device = torch.device('cuda:1')
    sd = StableDiffusion(device, opt.sd_version)

    imgs = sd.generate_twin_images([opt.prompt] * opt.n, [opt.negative] * opt.n, lam=opt.lam)  # [n,3,height,width,3]

    for i in tqdm(range(opt.n), desc='saving images'):
        img = view_images(imgs[i])
        img.save(f"out{i}.png")


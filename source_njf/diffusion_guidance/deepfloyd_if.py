import torch
import torch.nn.functional as F
from diffusers import IFPipeline, IFImg2ImgPipeline
from typing import Optional
from dataclasses import dataclass

@dataclass
class DiffusionConfig:
    """ Config for diffusion sds """
    # Model name
    pretrained_model_name_or_path: str = "DeepFloyd/IF-I-XL-v1.0"
    # Timestep sampling parameters
    min_step_percent: float = 0.02
    max_step_percent: float = 0.98
    # CFG guidance scale
    guidance_scale: float = 20.0
    # Whether or not to use half precision weights
    half_precision_weights: bool = True
    # Whether or not to use inpainting
    inpainting: bool = False

class DeepFloydIF:
    def __init__(self, cfg: DiffusionConfig=None):
        # Set device (cuda or cpu)
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        # If no config passed, use default config
        if cfg is None:
            cfg = DiffusionConfig()
        self.cfg = cfg

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        # Huggingface Diffusors pipeline
        self.pipe = IFPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            safety_checker=None,
            watermarker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            variant="fp16" if self.cfg.half_precision_weights else None,
            torch_dtype=self.weights_dtype,
        ).to(self.device)

        self.unet = self.pipe.unet.eval()

        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.scheduler = self.pipe.scheduler

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * self.cfg.min_step_percent)
        self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent)

        self.alphas: torch.FloatTensor = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

    def encode_prompt(self, prompt: str, negative_prompt: str = None):
        prompt_embeds, negative_embeds = self.pipe.encode_prompt(prompt, negative_prompt=negative_prompt)
        if self.cfg.guidance_scale > 1.0:
            self.prompt_embeds = torch.cat([negative_embeds, prompt_embeds])
        else:
            self.prompt_embeds = prompt_embeds
        return prompt_embeds, negative_embeds

    def mask_latents(self, background, selection, mask_image):
        mask_image = mask_image.unsqueeze(1).repeat_interleave(3, dim=1)
        return (1 - mask_image) * background + mask_image * selection

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: torch.FloatTensor,
        t: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
    ) -> torch.FloatTensor:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    def __call__(
        self,
        rgb_img: torch.FloatTensor, # Float[Tensor, "B C H W"]
        prompt_embeds: Optional[str] = None,
        negative_embeds: Optional[str] = None,
        **kwargs,
    ):
        batch_size = rgb_img.shape[0]

        # If prompt embeddings are passed, overwrite existing prompt embeddings
        if prompt_embeds is not None and negative_embeds is not None:
            self.prompt_embeds = torch.cat([negative_embeds, prompt_embeds])

        rgb_img = rgb_img * 2.0 - 1.0  # scale to [-1, 1] to match the diffusion range
        latents = F.interpolate(
            rgb_img, (64, 64), mode="bilinear", align_corners=False
        )

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=self.prompt_embeds.repeat_interleave(batch_size, dim=0),
            )  # (B, 6, 64, 64)

        # Perform classifier free guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred_uncond, _ = noise_pred_uncond.split(3, dim=1)
        noise_pred_text, predicted_variance = noise_pred_text.split(3, dim=1)
        noise_pred = noise_pred_text + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # w(t), sigma_t^2
        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        # clip grad for stable training (per threestudio)
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # Use reparameterization trick to avoid backproping gradient
        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        return {
            "loss_sds": loss_sds,
            "grad": grad,
            "grad_norm": grad.norm(),
            "target": target,
        }

class DeepFloydIF_Img2Img:
    def __init__(self, cfg: DiffusionConfig=None):
        # Set device (cuda or cpu)
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        # If no config passed, use default config
        if cfg is None:
            cfg = DiffusionConfig()
        self.cfg = cfg

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        # Huggingface Diffusors pipeline
        self.pipe = IFImg2ImgPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            variant="fp16" if self.cfg.half_precision_weights else None,
            torch_dtype=self.weights_dtype,
        ).to(self.device)
        self.pipe.enable_model_cpu_offload()

    def encode_prompt(self, prompt: str, negative_prompt: str = None):
        prompt_embeds, negative_embeds = self.pipe.encode_prompt(prompt, negative_prompt=negative_prompt)
        if self.cfg.guidance_scale > 1.0:
            self.prompt_embeds = torch.cat([negative_embeds, prompt_embeds])
        else:
            self.prompt_embeds = prompt_embeds
        return prompt_embeds, negative_embeds

    def mask_latents(self, background, selection, mask_image):
        mask_image = mask_image.unsqueeze(1).repeat_interleave(3, dim=1)
        return (1 - mask_image) * background + mask_image * selection

    def __call__(
        self,
        rgb_img: torch.FloatTensor, # Float[Tensor, "B C H W"]
        prompt_embeds: Optional[str] = None,
        negative_embeds: Optional[str] = None,
        rgb_as_latents=False,
        **kwargs,
    ):
        batch_size = rgb_img.shape[0]

        # If prompt embeddings are passed, overwrite existing prompt embeddings
        if prompt_embeds is not None and negative_embeds is not None:
            self.prompt_embeds = torch.cat([negative_embeds, prompt_embeds])

        generator = torch.manual_seed(1)

        # rgb_img = rgb_img * 2.0 - 1.0  # scale to [-1, 1] to match the diffusion range
        # latents = F.interpolate(
        #     rgb_img, (64, 64), mode="bilinear", align_corners=False
        # )

        # predict the full denoised image
        with torch.no_grad():
            target = self.pipe(image = rgb_img, prompt = self.prompt_embeds, generator = generator, output_type="pt").images
        loss_img = 0.5 * F.mse_loss(rgb_img, target, reduction="sum") / batch_size

        return {
            "loss_sds": loss_img,
            "target": target,
        }

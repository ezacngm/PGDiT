# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py


import math

import numpy as np
import torch
import torch as th
import enum
l2_loss = th.nn.MSELoss()
from .diffusion_utils import discretized_gaussian_log_likelihood, normal_kl
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import mean_squared_error as MSE
from torchvision.utils import save_image


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.
    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """
    This is the deprecated API for creating beta schedules.
    See get_named_beta_schedule() for the new library of schedules.
    """
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "warmup10":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == "warmup50":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        return get_beta_schedule(
            "linear",
            beta_start=scale * 0.0001,
            beta_end=scale * 0.02,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )
    elif schedule_name == "squaredcos_cap_v2":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class GaussianDiffusion_bvec_patch:
    """
    Utilities for training and sampling diffusion models.
    Original ported from this codebase:
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42
    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        num_frames,
        training=False
    ):
        self.training = training
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        self.num_frames = num_frames

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        ) if len(self.posterior_variance) > 1 else np.array([])

        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, raw_x=None, mask=None,bvecs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        # training = True

        B, C = x.shape[:2]
        
        assert t.shape == (B,)
        x = x.permute(0, 2, 1, 3, 4) #此时的x为x_t, x_start加噪后

        if not self.training and mask is not None:
            model_input = raw_x.permute(2, 0, 1, 3, 4) * (1-mask) + x.permute(2, 0, 1, 3, 4) * mask
            model_input = model_input.permute(1, 2, 0, 3, 4)
            model_output = model(model_input, t, bvecs, **model_kwargs)
        else:
            model_output = model(x, t, bvecs, **model_kwargs)
        # if not self.training and mask is not None:
        #     _m = (mask > 0).byte()
        #     _m = _m.unsqueeze(1).expand(-1, C, -1, -1, -1)  # [B, C, T, H, W]
        #     _m_comp = _m ^ 1
        #     _toflt = lambda Y: Y.to(dtype=x.dtype)
        #     _perm1 = lambda Y: Y.permute(2, 0, 1, 3, 4)
        #     _blend = lambda A, B, W1, W0: torch.einsum(
        #         'bcthw,bcthw->bcthw', (A * W1 + B * W0), torch.ones_like(A)
        #     )
        #     W1 = _toflt(_m_comp)
        #     W0 = _toflt(_m)
        #     Xp = _perm1(x)
        #     Rp = _perm1(raw_x)
        #     model_input = _blend(Xp, Rp, W1, W0)
        #     model_input = model_input.permute(1, 2, 0, 3, 4)
        #     model_output = model(model_input, t, bvecs, **model_kwargs)
        # else:
        #     model_output = model(x, t, bvecs, **model_kwargs)

        x = x.permute(0, 2, 1, 3, 4)
        if not self.training:
            model_output = model_output.permute(0, 2, 1, 3, 4)

        if isinstance(model_output, tuple):
            model_output, extra = model_output
        else:
            extra = None

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:]) #此出出错，记得diffusion 定义时改Training = False
            model_output, model_var_values = th.split(model_output, C, dim=1)
            min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(0, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_output)
        else:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            )
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "extra": extra,
            "eps": model_output
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def condition_mean(self, cond_fn, p_mean_var, x, t, raw_x, mask, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.
        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        with th.enable_grad():
            x = x.detach().requires_grad_(True)
            # p_mean_var["pred_xstart"] = p_mean_var["pred_xstart"].detach().requires_grad_(True)
            gradient = -th.autograd.grad(loss, x)[0]
        # new_mean = p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float().permute(0,2,1,3,4)
        new_mean = p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float().permute(0,1,2,3,4)

        # new_mean = p_mean_var["mean"].float() + gradient.float().permute(0,2,1,3,4)*1.2e6

        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.
        See condition_mean() for details on cond_fn.
        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, **model_kwargs)

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(x_start=out["pred_xstart"], x_t=x, t=t)
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        raw_x=None,
        mask=None,
        bvecs=None,
        recon=None,
        pde=True,
        mask_bvecs=None,
        sh_coeff_obs = None,
        manifold_proj = False,
            method = None
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model, x, t, clip_denoised=clip_denoised,
            denoised_fn=denoised_fn, model_kwargs=model_kwargs,
            raw_x=raw_x, mask=mask, bvecs=bvecs
        )
        # if recon is not None and t < 30:
        #     # Use Tweedie's formula: x_0 = (x_t - sqrt(1-alpha_t) * eps) / sqrt(alpha_t)
        #
        #     if isinstance(t, th.Tensor):
        #         alpha_t = self.alphas_cumprod[t.cpu().item()]
        #     else:
        #         alpha_t = self.alphas_cumprod[t]
        #

        #     if not isinstance(alpha_t, th.Tensor):
        #         alpha_t = th.tensor(alpha_t, device=x.device, dtype=x.dtype)
        #     sqrt_alpha_t = th.sqrt(alpha_t)
        #     sqrt_one_minus_alpha_t = th.sqrt(1 - alpha_t)
        #

        #     pred_xstart_uncond = out["pred_xstart"]

        #     recon_expanded = recon.unsqueeze(0).clamp(0, 1)
        #     mask_expanded = mask.unsqueeze(1)
        #
        #     pred_xstart_cond = (
        #             pred_xstart_uncond * (1 - mask_expanded) +
        #             recon_expanded.permute(0, 2, 1, 3, 4) * mask_expanded
        #     )
        #     guidance_scale=1.6
        #     pred_xstart = pred_xstart_uncond + guidance_scale * (pred_xstart_cond - pred_xstart_uncond)
        #
        #     if clip_denoised:
        #         pred_xstart = th.clamp(pred_xstart, 0, 1)
        #     out["mean"] = (
        #             sqrt_alpha_t * pred_xstart +
        #             sqrt_one_minus_alpha_t * out["eps"]
        #     )
        #
        # noise = th.randn_like(x)
        # nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        # sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise

        #
        #
        noise = th.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        if cond_fn is not None and t<50:
            x_copy = x.detach().clone()
            x_copy.requires_grad = True
            total_loss = 0.0
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = out["eps"]
            else:
                pred_xstart = self._predict_xstart_from_eps(x_t=x_copy, t=t, eps=out["eps"])
            if clip_denoised:
                pred_xstart = torch.clamp(pred_xstart, 0, 1)
            if recon is not None:
                recon = recon.unsqueeze(0).clamp(0,1)
                loss_obs = th.linalg.norm(((pred_xstart - recon.permute(0, 2, 1, 3, 4)).unsqueeze(1)))
                total_loss += loss_obs
                if method is not None:
                    method = method.unsqueeze(0).clamp(0, 1)
                    loss_method = th.linalg.norm(((pred_xstart - method.permute(0, 2, 1, 3, 4)).unsqueeze(1)))
                    total_loss = total_loss*0.0001 + loss_method

            else:
                from SH.spherical_harmonics_tensor import SphericalHarmonicProcessor_th as sh_tensor
                sf_func = sh_tensor(mask=mask, mask_bvecs=mask_bvecs)
                pred_xstart_progressive = pred_xstart.permute(1, 0, 2, 3, 4) * (mask) + raw_x.permute(2, 0, 1, 3, 4) * (1-mask)
                sh_coeff, recon = sf_func.run_subject(pred_xstart_progressive, bvecs)
                recon = recon.unsqueeze(0).clamp(0, 1)
                loss_obs = th.linalg.norm(((pred_xstart - recon.permute(0, 2, 1, 3, 4)) * (mask).unsqueeze(1)))
                total_loss += loss_obs
            if pde:
                from SH.spherical_harmonics_tensor import SphericalHarmonicProcessor_th as sh_tensor
                sf_func = sh_tensor(mask=mask,mask_bvecs=mask_bvecs)
                pred_xstart_progressive = pred_xstart.permute(1, 0, 2, 3, 4) * (mask) + raw_x.permute(2, 0, 1, 3,4) * (1 - mask)
                # sh_coeff= sf_func.get_sh_coeffs(pred_xstart.squeeze(0).permute(1,0,2,3),bvecs.squeeze(0))
                sh_coeff_progressive= sf_func.get_sh_coeffs(pred_xstart_progressive.squeeze(0).permute(1,0,2,3),bvecs.squeeze(0))
                k=0.6
                loss_sh = th.linalg.norm(((sh_coeff_progressive-sh_coeff_obs)))
                total_loss += k*loss_sh
            base_scale = 2e1  # Base multiplier (adjust as respaced steps)
            total_timesteps = self.num_timesteps if hasattr(self, 'num_timesteps') else 1000
            t_float = t.float() if not isinstance(t, th.Tensor) else t.float()
            # step_scale = base_scale * (1.0 + t_float / total_timesteps*2)
            step_scale = base_scale
            # Update the mean using the adaptive gradient update

            if manifold_proj: #add manifold constraints, coded by Chung et al. Neurips 2022, [https://github.com/hyungjin-chung/MCG_diffusion]
                gradient_norm = -th.autograd.grad(total_loss, x_copy)[0]
                shape = raw_x.shape
                # obs_proj = self.q_sample(raw_x.permute(2,0,1,3,4)*(1-mask), t, noise=th.randn_like(raw_x,device=raw_x.device).permute(2,0,1,3,4))
                obs_proj = self.q_sample(raw_x.permute(2,0,1,3,4), t, noise=nonzero_mask * noise)
                sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
                sample = sample *(mask) + gradient_norm.float().permute(0, 1, 2, 3, 4)*(mask) * step_scale + obs_proj*(1-mask)
                return {"sample": sample, "pred_xstart": out["pred_xstart"]}
            else:
                gradient = -th.autograd.grad(total_loss, x_copy)[0]
                out["mean"] = out["mean"].float() + gradient.float().permute(0, 1, 2, 3, 4) * step_scale
                sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
                return {"sample": sample, "pred_xstart": out["pred_xstart"]}
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        raw_x=None,
        mask=None,
        bvecs=None,
        mask_bvecs=None,
            method = None
    ):
        """
        Generate samples from the model.
        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            raw_x=raw_x,
            mask=mask,
            bvecs=bvecs,
            mask_bvecs=mask_bvecs,
            method = method
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        raw_x=None,
        mask=None,
        bvecs=None,
        mask_bvecs=None,
        sh_progressive=False,
            method = None
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if cond_fn is not None and progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        if cond_fn is not None and not sh_progressive:
            from SH.spherical_harmonics_tensor import SphericalHarmonicProcessor_th as sh_tensor

            sf_func = sh_tensor(mask=mask, mask_bvecs=mask_bvecs)
            sh_coeff, recon = sf_func.run_subject(raw_x, bvecs)
            recon = recon.clamp(0,1) * mask.permute(1,0,2,3) + raw_x.squeeze(0) * (1 - mask.permute(1,0,2,3))
        else:
            recon = None
            sh_coeff = None
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            out = self.p_sample(
                model,
                img,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                raw_x=raw_x,
                mask=mask,
                bvecs=bvecs,
                recon=recon,
                mask_bvecs=mask_bvecs,
                sh_coeff_obs = sh_coeff,
                method=method
            )
            yield out
            img = out["sample"]


    def _vb_terms_bpd(
            self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None, mask=None,bvecs=None
    ):
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, posterior_variance, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )

        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs, raw_x=x_start, mask=mask,bvecs=bvecs
        )#x_t(1,4,90,40,40),x_start(1,4,90,40,40),mask(1,90,40,40)
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, mask=None,bvecs=None):
        """
        Compute training losses for a single timestep.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        x_start = x_start.permute(0, 2, 1, 3, 4)


        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)#隐空间noisy_gt  (1,4,90,40,40)

        terms = {}

        if self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            x_t = x_t.permute(0, 2, 1, 3, 4) #隐空间noisy_gt  (1,90,4,40,40)
            # mask, B T H W (1,90,40,40)
            model_input = x_start.permute(1, 0, 2, 3, 4) * (1-mask) + x_t.permute(2, 0, 1, 3, 4) * mask
            model_input = model_input.permute(1, 2, 0, 3, 4) #
            model_output = model(model_input, t, bvecs, **model_kwargs)

            x_t = x_t.permute(0, 2, 1, 3, 4)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                model_output = model_output.permute(0, 2, 1, 3, 4)
                
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                combined = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,#(1,4,90,40,40)
                    x_t=x_t,#(1,4,90,40,40)
                    t=t,
                    clip_denoised=True,
                    mask=mask,#(1,90,40,40)
                    bvecs=bvecs
                )
                terms["vb"] = combined["output"]
                x0_predicted = combined["pred_xstart"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape #(1,4,90,40,40)

            terms["mse"] = mean_flat(((target - model_output).permute(1, 0, 2, 3, 4) * mask).permute(1, 0, 2, 3, 4)  ** 2)
            terms["mae"] = mean_flat(((target - model_output).permute(1, 0, 2, 3, 4) * mask).permute(1, 0, 2, 3, 4) )
            # terms["mse"] = l2_loss(target,model_output)

            assert x0_predicted.shape == x_start.shape
            terms["xstart_mae"] = mean_flat(((x0_predicted - x_start).permute(1,0,2,3,4)*mask).permute(1,0,2,3,4).abs())
            # terms["xstart_mae"] = mean_flat(((x0_predicted - x_start).permute(1,0,2,3,4)).permute(1,0,2,3,4).abs())
            terms["xstart_mse"] = mean_flat(((x0_predicted - x_start).permute(1, 0, 2, 3, 4) * mask).permute(1, 0, 2, 3, 4)  ** 2)
            lambda_ang = 0.1
            x0_pred = x0_predicted  # (B,C,T,H,W)
            x0_gt = x_start  # (B,C,T,H,W)

            terms["ang"] = lambda_ang * angular(x0_pred, x0_gt, bvecs, mask,losstype='l1')  # scalar

            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms, x0_predicted, x_start



def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + th.zeros(broadcast_shape, device=timesteps.device)


def angular(x_pred, x_gt, bvecs, mask=None, losstype='l1'):

    B, C, T, H, W = x_pred.shape
    # (B,T,C,H,W)
    x_pred_t = x_pred.permute(0, 2, 1, 3, 4)
    x_gt_t   = x_gt.permute(0, 2, 1, 3, 4)

    mu_pred = x_pred_t.mean(dim=1, keepdim=True)
    mu_gt   = x_gt_t.mean(dim=1, keepdim=True)

    res_pred = x_pred_t - mu_pred                     # (B,T,C,H,W)
    res_gt   = x_gt_t   - mu_gt
    bvecs_n  = torch.nn.functional.normalize(bvecs, dim=-1)  # (B,T,3)
    proj_pred = (res_pred[:, :, :3] *
                 bvecs_n[..., None, None]).sum(dim=2)         # (B,T,H,W)
    proj_gt   = (res_gt  [:, :, :3] *
                 bvecs_n[..., None, None]).sum(dim=2)
    diff = (proj_pred - proj_gt)
    if losstype == 'l1':
        ang= diff.abs()              # (B,T,H,W)
    elif losstype == 'l2':
        ang= diff.pow(2)
    elif losstype == 'huber':
        ang = torch.where(diff.abs() < 0.01,0.5*diff.pow(2)/0.01, diff.abs()-0.5*0.01)
    if mask is not None:
        ang = ang * mask
    return ang.mean()                              # scalar

# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from . import gaussian_diffusion as gd
from . import gaussian_diffusion_bvec as gd_bvec
from . import gaussian_diffusion_bvec_patch as gd_bvec_patch
from .respace import SpacedDiffusion, space_timesteps,SpacedDiffusion_bvec,SpacedDiffusion_bvec_patch


def create_diffusion(
    timestep_respacing,
    noise_schedule="linear", 
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000,
    num_frames=90,
    training=True
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        num_frames=num_frames,
        training=training
        # rescale_timesteps=rescale_timesteps,
    )
def create_diffusion_bvec(
    timestep_respacing,
    noise_schedule="linear",
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000,
    num_frames=90,
    training=True
):
    betas = gd_bvec.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd_bvec.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd_bvec.LossType.RESCALED_MSE
    else:
        loss_type = gd_bvec.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion_bvec(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd_bvec.ModelMeanType.EPSILON if not predict_xstart else gd_bvec.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd_bvec.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd_bvec.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd_bvec.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        num_frames=num_frames,
        training=training
        # rescale_timesteps=rescale_timesteps,
    )

def create_diffusion_bvec_patch(
    timestep_respacing,
    noise_schedule="linear",
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000,
    num_frames=90,
    training=True
):
    betas = gd_bvec_patch.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd_bvec_patch.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd_bvec_patch.LossType.RESCALED_MSE
    else:
        loss_type = gd_bvec_patch.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion_bvec_patch(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd_bvec_patch.ModelMeanType.EPSILON if not predict_xstart else gd_bvec_patch.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd_bvec_patch.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd_bvec_patch.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd_bvec_patch.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        num_frames=num_frames,
        training=training
        # rescale_timesteps=rescale_timesteps,
    )

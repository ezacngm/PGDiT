import yaml
import random
import torchvision
import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '45400'
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
print(os.environ["CUDA_VISIBLE_DEVICES"])
print("DiT training on GPU: ",os.environ["CUDA_VISIBLE_DEVICES"],"master Port: ",os.environ["MASTER_PORT"])
import wandb
from tqdm import tqdm
import torch
import sys
import torch.nn.functional as F
import argparse
from itertools import islice
import numpy as np
from diffusion import create_diffusion_bvec_patch
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from models.models import PGDiT_models
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from mask_generator import MaskGenerator_progressive
from torch.utils.data.dataloader import DataLoader
from dataset import HCPDataset_Cond
from collections import OrderedDict
from torchvision.utils import make_grid, save_image
from utils.metric import metric_tensor as get_metric
import einops
from copy import deepcopy
from glob import glob
import nibabel as nib
from time import time
import logging
from utils.reshaper import sliding_windows_cut,sliding_windows_stick
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import mean_squared_error as MSE
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device_ids = [0,1]
#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()
def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(0, len(tensor.shape))))



def cond_fn_measurement(x, t, bvecs, ground_truth, mask,x_0_hat, **kwargs):
    """
    Guidance function based on a measurement (MSE) between the generated sample and ground truth.

    Args:
        x (Tensor): The current sample from the diffusion process.
        t (Tensor): The current timestep (may be unused in this example).
        bvecs: Additional argument placeholder.
        ground_truth (Tensor): The ground truth image or signal.
        mask (Tensor): A binary mask indicating regions where the ground truth is known.

    Returns:
        Tensor: The gradient of the measurement loss with respect to x.
    """
    import torch as th
    with th.enable_grad():  # ensure gradients are enabled
        # Ensure x is tracked for gradients.
        x = x.detach().requires_grad_(True)

        # Compute the measurement loss (MSE) on the masked region.
        loss = th.linalg.norm(((x - ground_truth) * mask))

        # Compute the gradient of the loss with respect to x.
        grad = th.autograd.grad(loss, x)[0]
    # Return the negative gradient to guide the sample toward lower error.
    return -grad
#################################################################################
#                                  Sampling Loop                                #
#################################################################################
def train(args, wandb_on=False):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    dataset_config = config['dataset_params']
    dit_config = config["dit_params"]
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']

    # Setup DDP:
    rank_in=0
    world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    print("rank: ",rank_in,"world_size: ",world_size)
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank_in, world_size=world_size)
    # assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # ------------------------Create output directories--------------------------
    if not os.path.exists(train_config['vae_name']):
        os.mkdir(train_config['vae_name'])
    roor_dir = "/mnt/siat205_disk2/nanmu/experiments/PGDiT/"
    vae_dir = os.path.join(roor_dir,train_config["vae_name"])
    dit_output_dir = os.path.join(vae_dir, train_config["dit_name"])
    os.makedirs(dit_output_dir,exist_ok=True)
    # ### init wandb
    if wandb_on:
        # wandb.init(project="vqvae",name=train_config["vae_name"],id="oc8ead1u",resume="must")
        wandb_logger = wandb.init(project="qspace_DiT_cond", name=train_config["vae_name"])
        wandb.config.update(config)
        print(wandb.run.id)
    # --------------------------Set the desired seed value # ----------------------
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    #############################


    # ----------------------------------Instantiate the model------------------------
    # Load model:
    additional_kwargs = {'num_frames': dataset_config["q_space_directions"],
                         'mode': 'video'}
    if dit_config["bvec_emb"] == "mlp" :
        from models.models import PGDiT_models
    model = PGDiT_models[dit_config["model"]](
        input_size=dataset_config["im_size"],
        num_classes=dit_config["num_classes"],
        in_channels=1,
        **additional_kwargs
    ).to(device)
    # Note that parameter initialization is done within the DiT constructor


    print('Loaded dit model')
    model_path = dit_config["finetune_ckpt"]

    if os.path.exists(model_path):
        model.load_state_dict(
            torch.load(model_path,
                       map_location=torch.device(f'cuda:{device}'),weights_only=True))
        print(f'Loaded dit checkpoint at{model_path}')
    else:
        print('?????????????No dit checkpoint')
    model = DDP(model, device_ids=[rank])

    model.eval()

    # ----------------------create ddpm----------------------------------
    diffusion = create_diffusion_bvec_patch(str(dit_config["num_sampling_respace"]),
                                    noise_schedule="linear",
                                    use_kl=False,
                                    sigma_small=False,
                                    predict_xstart=False,
                                    learn_sigma=True,
                                    rescale_learned_sigmas=False,
                                    diffusion_steps=1000,
                                    num_frames=dataset_config["q_space_directions"],
                                    training=False)

    # ----------------------------------Instantiate the optimiser------------------------
    # opt = torch.optim.AdamW(model.parameters(), lr=train_config["dit_lr"], weight_decay=0)
    # logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    # ---------------------------------------Create the dataset-------------------------------

    HCP_dataset = HCPDataset_Cond(
        data_folder=dataset_config["data_folder"],
        train=True,
        sequence_length=dataset_config["q_space_directions"],
        resolution=dataset_config["im_size"],
        shell = dataset_config["shell"],
        patch=True
    )
    train_sampler =  DistributedSampler(
        HCP_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )

    train_dataloader = DataLoader(HCP_dataset, batch_size=train_config['ldm_batch_size'],
                             shuffle=False,sampler = train_sampler,
                             pin_memory=True,
                             drop_last=True, num_workers=8)

    # Prepare models for training:
    model.eval()  # important! This enables embedding dropout for classifier-free guidance


    # ------------------------Create output directories--------------------------
    if not os.path.exists(train_config['vae_name']):
        os.mkdir(train_config['vae_name'])

    num_epochs = train_config['autoencoder_epochs']

    # L1/L2 loss



    # This is for accumulating gradients incase the images are huge
    # And one cant afford higher batch sizes
    acc_steps = train_config['autoencoder_acc_steps']
    image_save_steps = train_config['autoencoder_img_save_steps']
    model_save_steps = train_config['model_save_steps']
    img_save_count = 0
    sub_index = 0
    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    with torch.no_grad():
        # --------------------------run testing-------------------------

        recon_losses = []
        perceptual_losses = []
        losses = []
        ssims = []
        psnrs = []
        rmses = []
        sub_recon = []
        sub_gt = []
        # print("train_step:", train_steps, "difficulty(masking_ratio): ", (5 + 0.5 * (train_steps // 5000)))
        # progress_bar = tqdm(a_loader, desc='test',ncols=150)
        sampling = [4, 7, 17, 37, 45, 52]

        pattern = [sampling]
        shells = [1]
        # sampling = np.random.choice(np.arange(0, 90), size=10, replace=False)

        for subject in [210]:

            for shell in shells:

                test_dataset = HCPDataset_Cond(
                    data_folder=dataset_config["data_folder"],
                    train=False,
                    sequence_length=dataset_config["q_space_directions"],
                    resolution=dataset_config["im_size"],
                    shell=shell,
                    kth_subject=subject,
                    patch=False
                )

                test_dataloader = DataLoader(test_dataset, batch_size=train_config['ldm_batch_size'], shuffle=False,
                                             drop_last=False, num_workers=4)
                for sampling in pattern:
                    slice_id = 0
                    a_loader = islice(test_dataloader, slice_id, None)
                    progress_bar = tqdm(a_loader, desc='test', ncols=150)
                    for im,bvecs in progress_bar:

                        orginal_shape = im.squeeze().shape
                        image_save_steps+=1
                        im = im.float().squeeze(0).unsqueeze(1)
                        im = im[:].to(device)
                        if im.max().item() <= 0:
                            print(f'slice {slice_id} is all zero and skipped')
                            slice_id+=1
                            continue
                        bvecs = bvecs.float().to(device)
                        patches, coords = sliding_windows_cut(im.squeeze(), patch_size=(63,63))
                        merged_image = []
                        for patch in patches:
                            im_input = patch.unsqueeze(1)
                            im_gt = patch.unsqueeze(1).float().to(device)

                            x_latents = einops.repeat(im_input, 't c h w -> b t c h w', b=1)
                            B, T, C, H, W = x_latents.shape

                            # ------------set spatial temporal mask generater
                            mask_generator = MaskGenerator_progressive((T, H, W),sampling=sampling)
                            mask,bvec_mask = mask_generator(B, device, idx=6)
                            bvecs_masked = bvecs * (1-bvec_mask)

                            #----------------------------samplling using posterier sample loop--------------
                            z = torch.randn_like(x_latents, device=device) # 隐空间形状torch.Size([1, 90, 4, 40, 40])
                            sample_fn = model.forward
                            # Sample images:
                            z = z.permute(0, 2, 1, 3, 4) #要改成(1,4,90,40,40)????不明所以
                            # print("z shape is changed to be:", z.shape)

                            samples = diffusion.p_sample_loop(
                                sample_fn, z.shape, z, clip_denoised=True, progress=True, device=device,
                                raw_x=x_latents, mask=mask, bvecs=bvecs,cond_fn=None,mask_bvecs=bvec_mask,
                                # method = recon[:,slice_id:slice_id+1],
                            )

                            samples_copy = samples
                            sub_recon.append(samples_copy)
                            sub_gt.append(im_gt)
                            mask_latent = mask

                            # ------------------------------samples shape modification-----------------------------------

                            samples = samples.permute(1, 0, 2, 3, 4) * mask + x_latents.permute(2, 0, 1, 3, 4) * (1 - mask)

                            samples = samples.permute(1, 2, 0, 3, 4)
                            samples = samples.reshape(-1, 1, dataset_config["im_size"],dataset_config["im_size"] )
                            #------------------latent image saving logic-----------------------------------
                            samples_latents_copy = samples.clamp(0,1)

                            save_image(samples_latents_copy, f"{dit_output_dir}/output.png",nrow=30,normalize=True, value_range=(0, 1))

                            x_latents_copy = x_latents.squeeze(0)

                            merged_image.append(samples_latents_copy.squeeze())
                            save_image(x_latents_copy, f"{dit_output_dir}/gt.png", nrow=30,normalize=True, value_range=(0, 1))

                            diff_latent = x_latents_copy - samples_latents_copy
                            save_image(diff_latent.abs(), f"{dit_output_dir}/diff.png", nrow=30,normalize=True, value_range=(0, 0.5))
                            print("done for a patch")
                            #
                            #
                            # # print("ssim:", ssim_value, "psnr:", psnr_value, "rmse:", rmse_value)
                            # # print("this image is done sampling")
                            # ssims.append(ssim_value)
                            # if str(psnr_value)!='inf':
                            #     psnrs.append(psnr_value)
                            # rmses.append(rmse_value)
                            #
                            # decoded_samples = samples
                            # mask = mask.unsqueeze(0).repeat(1, 1, 1, 1, 1).permute(1, 2, 0, 3, 4)
                            #
                            # im_input = im_input.reshape(-1, args.num_frames, im_input.shape[-3], im_input.shape[-2], im_input.shape[-1])
                            # im_input = im_input * (1 - mask)
                            # masked_in_and_output = torch.cat([im_input.squeeze(0), decoded_samples], dim=0)
                            #
                            # #-------------------------------------image saving logic------------------------------------
                            # save_image(masked_in_and_output.reshape(-1, 1, masked_in_and_output.shape[-2],
                            #                                         masked_in_and_output.shape[-1]),
                            #            "/mnt/siat205_disk2/nanmu/experiments/PGDiT/monai_baseline_latentdim4/input_output.png", nrow=30, normalize=True, value_range=(0, 1))50
                        merged = torch.stack(merged_image,dim=0)
                        merged = sliding_windows_stick(merged, coords, output_shape=orginal_shape, patch_size=(63, 63))
                        save_image(merged.unsqueeze(1), f"{dit_output_dir}/output.png", nrow=30,
                                   normalize=True, value_range=(0, 1))
                        exp_dir = '/mnt/experiments/'
                        sub_dir = os.path.join(exp_dir, f'shell_{shell}',f'sub_{subject:03d}',f'sub_{subject:03d}_sampling{len(sampling)}_test')
                        os.makedirs(sub_dir, exist_ok=True)
                        slice_path = os.path.join(sub_dir,f'slice_{slice_id:03d}.npy')
                        np.save(slice_path,merged.cpu().numpy())
                        ssim_value, _ = SSIM(im.squeeze().cpu().numpy(),merged.squeeze().cpu().numpy(), data_range=1, full=True)
                        psnr_value = PSNR(im.squeeze().cpu().numpy(),merged.squeeze().cpu().numpy(), data_range=1)
                        rmse_value = np.sqrt(MSE(im.squeeze().cpu().numpy(), merged.squeeze().cpu().numpy()))
                        slice_id += 1


        cleanup()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae training')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    parser.add_argument("--global_batch_size", type=int, default=1)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)  # Set higher for better results! (max 1000)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=90)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    try:
        train(args)
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught, cleaning up...")
        cleanup()
        sys.exit(0)

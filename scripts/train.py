import yaml
import random
import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '44200'
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
print(os.environ["CUDA_VISIBLE_DEVICES"])
print("DiT training on GPU: ",os.environ["CUDA_VISIBLE_DEVICES"],"master Port: ",os.environ["MASTER_PORT"])
os.environ['WANDB_API_KEY'] = '1e96743987e87d422f9a3c81ad232fcc0c168f12'

import wandb
from tqdm import tqdm
import argparse
import numpy as np
import sys
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader


from diffusion import create_diffusion_bvec_patch
from mask_generator import MaskGenerator_progressive
from torch.cuda.amp import GradScaler, autocast
from generative.losses import PatchAdversarialLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator


from dataset import HCPDataset_Cond
from collections import OrderedDict
from torchvision.utils import make_grid, save_image
from utils.metric import metric_tensor as get_metric
import einops
from copy import deepcopy
from glob import glob
from time import time
import logging


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device_ids = [0,1]
#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


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


#################################################################################
#                                  Training func                               #
#################################################################################

def train(args, wandb_on=False,enable_ema=True):
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
    wandb_on = train_config['wandb_on']

    # Setup DDP:
    rank_in=0
    world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    print("rank: ",rank_in,"world_size: ",world_size)
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank_in, world_size=world_size)
    rank=dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")



    # ------------------------Create output directories--------------------------
    roor_dir = "/mnt/siat205_disk2/nanmu/experiments/PGDiT/"
    vae_dir = os.path.join(roor_dir,train_config["vae_name"])
    dit_output_dir = os.path.join(vae_dir, train_config["dit_name"])
    os.makedirs(dit_output_dir,exist_ok=True)
    # ### init wandb
    if wandb_on:
        wandb_logger = wandb.init(
            project="PGDiT",
            name=train_config["vae_name"],
            mode="offline",
            dir=dit_output_dir  # ← 将本地日志写入此目录
        )
        wandb.config.update(config, allow_val_change=True)
        print("WandB 日志目录：", wandb.run.dir)
    train_steps = 0
    # --------------------------Set the desired seed value # ----------------------
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    #############################
    # ----------------------create ddpm

    diffusion = create_diffusion_bvec_patch(timestep_respacing="",
                                      noise_schedule="linear",
                                      use_kl=False,
                                      sigma_small=dit_config['sigma_small'],
                                      predict_xstart=dit_config['predict_xstart'],
                                      learn_sigma=dit_config['learn_sigma'],
                                      rescale_learned_sigmas=False,
                                      diffusion_steps=1000,
                                      num_frames=dataset_config["q_space_directions"],
                                      training=True)




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
        learn_sigma=dit_config['learn_sigma'],
        **additional_kwargs
    ).to(device)
    # Note that parameter initialization is done within the DiT constructor
    if enable_ema:
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        print('ema model loaded')

    print('Loaded dit model')
    if dit_config['finetune_ckpt']:
        model_path = dit_config['finetune_ckpt']
        if not os.path.exists(model_path):
            raise FileNotFoundError('No checkpoint found at {},fail to start fine-tuning'.format(model_path))
        model.load_state_dict(torch.load(model_path,map_location=torch.device(f'cuda:{device}'),weights_only=True))
        print(f'Loaded dit checkpoint for fine-tuning at {model_path}')
    model = DDP(model, device_ids=[rank])
    model.train()
    # ----------------------------------Instantiate the optimiser------------------------

    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20, eta_min=1e-6)
    # ---------------------------------------Create the dataset-------------------------------
    scaler = GradScaler()
    HCP_dataset = HCPDataset_Cond(
        data_folder=dataset_config["data_folder"],
        train=True,
        sequence_length=dataset_config["q_space_directions"],
        resolution=dataset_config["im_size"],
        shell=dataset_config['shell'],
        patch = True
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

    test_dataset = HCPDataset_Cond(
        data_folder=dataset_config["data_folder"],
        train=False,
        sequence_length=dataset_config["q_space_directions"],
        resolution=dataset_config["im_size"]
    )

    test_dataloader = DataLoader(test_dataset, batch_size=train_config['autoencoder_batch_size'], shuffle=False,
                                 drop_last=True, num_workers=4)

    # Prepare models for training:
    if enable_ema:
        update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
        ema.eval()  # EMA model should always be in eval mode
    model.train()  # important! This enables embedding dropout for classifier-free guidance


    # ------------------------Create output directories--------------------------

    num_epochs = train_config['autoencoder_epochs']

    # L1/L2 loss
    l1_loss = torch.nn.L1Loss(reduction="mean")
    l2_loss = torch.nn.MSELoss()


    # This is for accumulating gradients incase the images are huge
    # And one cant afford higher batch sizes
    acc_steps = train_config['autoencoder_acc_steps']
    image_save_steps = train_config['autoencoder_img_save_steps']
    model_save_steps = train_config['model_save_steps']
    img_save_count = 0
    # Variables for monitoring/logging purposes:

    log_steps = 0
    running_loss = 0
    start_time = time()
    recon_losses = []
    losses = []
    mse = []
    vb = []
    xstart_mse = []
    xstart_mae = []
    ang_l1 = []
    ssims = []
    psnrs = []
    rmses = []
    # --------------------------run training-------------------------
    for epoch_idx in range(num_epochs):
        train_sampler.set_epoch(epoch_idx)

        # print("train_step:", train_steps, "difficulty(masking_ratio): ", (5 + 0.5 * (train_steps // 5000)))

        progress_bar = tqdm(train_dataloader, desc='train',ncols=150)
        for im,bvecs in progress_bar:
            im = im.float().squeeze(0).unsqueeze(1) #(90,1,160,160)
            im = im[:].to(device)
            bvecs = bvecs.float().to(device)
            if im.max().item() <= 0:
                continue
            im_input = im
            im_gt = im.float().to(device)

            # --------------------------------Fetch autoencoders output(reconstructions)
            # with torch.no_grad():
            #     # Map input images to latent space + normalize latents:
            #     x_latents,_ = vae.module.encode(im_input) #隐空间形状 (90,4,40,40)
            x_latents = einops.repeat(im_input, 't c h w -> b t c h w', b=1)
            B, T, C, H, W = x_latents.shape

            # ------------set spatial temporal mask generater
            mask_generator = MaskGenerator_progressive((T, H, W),mask_ratio=0.7)
            # mask_generator.set_difficulty(8/7)
            mask,bvec_mask = mask_generator(B, device, idx=5)
            bvecs_masked = bvecs * (1-bvec_mask)



            t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)

            # loss_dict,_, _ = diffusion.training_losses(model, x_latents, t, mask=mask,bvecs=bvecs)

            with autocast():
                loss_dict,x0_predicted, x_start = diffusion.training_losses(model, x_latents, t, mask=mask,bvecs=bvecs)
            ssim_value,psnr_value,rmse_value = get_metric(x_start.squeeze().unsqueeze(1).detach().cpu(),x0_predicted.squeeze().unsqueeze(1).detach().cpu())
            ssims.append(ssim_value)
            psnrs.append(psnr_value)
            rmses.append(rmse_value)
            loss = loss_dict["loss"].mean()
            mse_loss = loss_dict["mse"].mean()
            vb_loss = loss_dict["vb"].mean()
            angular_l1_loss = loss_dict['ang'].mean()
            xstart_mse_loss = loss_dict["xstart_mse"].mean()
            xstart_mae_loss = loss_dict["xstart_mae"].mean()
            losses.append(loss.item())
            mse.append(mse_loss.item())
            ang_l1.append(angular_l1_loss.item())
            vb.append(vb_loss.item())
            xstart_mse.append(xstart_mse_loss.item())
            xstart_mae.append(xstart_mae_loss.item())

            opt.zero_grad()  # <<< unchanged
            scaler.scale(loss).backward()  # <<< use scaler.scale(...)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)  # <<< ...and scaler.step(...)
            scaler.update()  # <<< ...then scaler.update()
            if enable_ema:
                update_ema(ema, model.module)
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            log_dict = {"total loss":np.mean(losses),
                        "mse":np.mean(mse),
                        "vb":np.mean(vb),
                        "xstart_mse":np.mean(xstart_mse),
                        "xstart_mae":np.mean(xstart_mae),
                        'angular_l1':np.mean(ang_l1)
                        }
            if wandb_on:
                wandb_logger.log(log_dict,
                                 step=train_steps
                                 )

            progress_bar.set_postfix(
                loss=np.mean(losses),
                xstart_mse=np.mean(xstart_mse),
                xstart_mae=np.mean(xstart_mae),
                ang_l1 = np.mean(ang_l1),
                ssim=np.mean(ssims),
                psnr=np.mean(psnrs),
                rmse=np.mean(rmses),
            )
            # gt_predicted_path = f"/home/munan/PGDiT/medidiffvae_dit/running_gt_predicted/gt_predicted_{log_steps:05d}.png"
            # save_image(gt_predicted,gt_predicted_path,nrow=30,normalize=True,value_range=(0,1))
            if train_steps % model_save_steps == 0 and train_steps > 0:
                # if rank == 0:
                model_save_path = os.path.join(dit_output_dir, f"dit_{train_steps:06d}")
                os.makedirs(model_save_path, exist_ok=True)
                checkpoint = {
                    "model": model.module.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args,
                    "step": train_steps,
                }
                save_path_dict = {
                    "model": os.path.join(model_save_path, f"dit_{train_steps:06d}.pth"),
                    "opt": os.path.join(model_save_path, f"dit.opt"),
                    "args": os.path.join(model_save_path, f"dit_{train_steps:06d}.args"),
                    "step": train_steps,
                    "ema": os.path.join(model_save_path, f"dit_{train_steps:06d}_ema.pth"),
                }
                if enable_ema and ema is not None:
                    checkpoint["ema"] = ema.state_dict()
                    torch.save(checkpoint["ema"], save_path_dict['ema'])
                torch.save(checkpoint['model'], save_path_dict['model'])
                torch.save(checkpoint['opt'], save_path_dict['opt'])
                print(f"[Checkpoint] Saved at step {train_steps}")
                # dist.barrier()
        scheduler.step()  # <---  scheduler 在這裡更新！
    cleanup()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae training')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--f", type=str, default=None)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--global_batch_size", type=int, default=1)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=16)  # Set higher for better results! (max 1000)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--ckpt", type=str, default="model.pt",
                        help="Optional path to a PGDiT checkpoint.")
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    try:
        train(args)
        exit_code = 0
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught, cleaning up...")
        exit_code = 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

    sys.exit(exit_code)


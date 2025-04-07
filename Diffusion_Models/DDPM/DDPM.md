#DDPM

### Model Architecture

     GraphImageDenoiser(

       (encoder): GraphEncoder(
           (conv1): GCNConv(3, 128)
           (conv2): GCNConv(128, 64)
           (conv3): GCNConv(64, latent_dim)
       )

       (decoder): GraphDecoder(
           (lin1): Linear(in_features=latent_dim, out_features=64, bias=True)
           (lin2): Linear(in_features=64, out_features=128, bias=True)
       )

       (ddpm): UNet2DModel(
           sample_size=128,
           in_channels=3,
           out_channels=3,
           layers_per_block=2,
           block_out_channels=(64, 128, 256, 512),
           down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
           up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
       )

       (scheduler): DDPMScheduler(
           num_train_timesteps=1000
       )

       (loss_fn_alex): LPIPS(net='alex')
       (loss_fn_vgg): LPIPS(net='vgg')

       (loss_fn): CharbonnierLoss(eps=1e-3)
       (contrastive_loss): ContrastiveGraphLoss(margin=1.0)

       )

### Results

Epoch 5 | Train Loss: 0.1726 | Val Loss: 0.1719

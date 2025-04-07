#Unet2DConditional

### Model Architecture

      Denoiser(

          (conv1): GCNConv(3, 64)
          (conv2): GCNConv(64, graph_hidden)

          (proj): Linear(in_features=graph_hidden, out_features=proj_dim, bias=True)
      
          (unet): UNet2DConditionModel(
              sample_size=64,
              in_channels=3,
              out_channels=3,
              layers_per_block=1,
              block_out_channels=(32, 64, 128, 256),
              down_block_types=(
                  "DownBlock2D",
                  "CrossAttnDownBlock2D",
                  "CrossAttnDownBlock2D",
                  "DownBlock2D"
              ),
              up_block_types=(
                  "UpBlock2D",
                  "CrossAttnUpBlock2D",
                  "UpBlock2D",
                  "UpBlock2D"
              ),
              cross_attention_dim=proj_dim
          )
      
          (timestep): Buffer(tensor of shape (1,), dtype=torch.long, value=0)

    )

### Results

Epoch 5: Train Loss 0.0001, Val Loss 0.0001

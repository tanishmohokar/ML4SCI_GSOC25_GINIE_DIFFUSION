# 🧠 Graph-Conditioned Image Denoising with Diffusion Models

---

This repository contains two different architectures designed to denoise jet images by incorporating graph-based representations of point clouds from high-energy physics data. The two approaches are:

  - Graph-DDPM: Combines a Graph Autoencoder with a Denoising Diffusion Probabilistic Model (DDPM).

  - GCN-Conditioned UNet: Uses graph features as conditioning input via cross-attention in a UNet-based architecture.

---

## 📊 Datasets

The models are designed to work with jet images converted from particle data. Each image has:

- 3 channels: tracking info, ECAL, and HCAL

- Accompanying point cloud graph data: each non-zero pixel becomes a     node, edges are defined via spatial or physics-based proximity



## 🔍 Motivation

Jet images derived from particle detectors (e.g., LHC) are often noisy due to detector resolution and overlapping events. Traditional denoising methods ignore the underlying particle-level structure.

This project introduces graph representations of jets as conditioning inputs to denoising networks, combining:

  - Point cloud intelligence via GCNs

  - Powerful generative capabilities of DDPMs and UNets

  - Cross-modality learning from graph → image

## 🚀 Architectures Overview

### 1️⃣ Graph-DDPM

This model builds a joint denoising pipeline using:

  - A Graph Encoder using GCNConv layers to extract graph features from particle data.

  - A Graph Decoder that attempts to reconstruct the adjacency matrix, aiding unsupervised graph learning.

  - A UNet2DModel used for image denoising via diffusion.

  - A Contrastive Loss on the graph embeddings to preserve graph topology.

  - A Charbonnier Loss for denoising, combined with LPIPS loss options.

### 2️⃣ GCN-Conditioned UNet 

This model implements a GCN-based feature extractor which conditions the UNet via cross-attention:

  - Uses two layers of GCN (GCNConv) to obtain a graph-level embedding via global mean pooling.

  - This embedding is projected to match UNet’s cross-attention dimensionality (512 by default).

  - A UNet2DConditionModel uses these features as guidance to denoise images with skip connections.


## 📈 Losses Used

  - Charbonnier Loss: For robust image reconstruction.

  - Contrastive Loss: Pulls together connected nodes and pushes apart random ones in embedding space.

## 📏 Evaluation Metrics

1. Charbonnier Loss
- A smooth, robust alternative to Mean Squared Error (MSE), less sensitive to outliers. It helps stabilize training, especially for image denoising tasks.

2. LPIPS (Learned Perceptual Image Patch Similarity)
- A perceptual similarity metric that uses deep network features (like AlexNet or VGG) to evaluate how visually similar two images are, closer to human perception than pixel-wise losses.

3. Contrastive Graph Loss
- Encourages connected nodes in a graph to have similar embeddings, while pushing apart randomly sampled unconnected pairs. Helps the model learn meaningful graph structures.

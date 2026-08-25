import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class PerceptualAttackLoss:
    def __init__(self, weight=1.0, feature_scale=1.0, visualize=False, save_dir=None):
        self.weight = float(weight)
        self.feature_scale = float(feature_scale)
        self.visualize = visualize
        self.save_dir = save_dir
        self._feature_extractor = self._build_feature_extractor()

    def _build_feature_extractor(self):
        return nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def _preprocess_sequence(self, sequence):
        if sequence.dim() == 5:
            # Support both [B, S, C, H, W] and [B, C, S, H, W] layouts.
            if sequence.shape[1] <= 4 and sequence.shape[2] > 4:
                frames = sequence.permute(1, 0, 2, 3, 4).contiguous().reshape(-1, sequence.shape[2], sequence.shape[3], sequence.shape[4])
            else:
                frames = sequence.permute(2, 0, 1, 3, 4).contiguous().reshape(-1, sequence.shape[1], sequence.shape[3], sequence.shape[4])
        elif sequence.dim() == 4:
            frames = sequence.reshape(-1, 1, sequence.shape[-2], sequence.shape[-1])
        elif sequence.dim() == 3:
            frames = sequence.unsqueeze(1)
        else:
            raise ValueError(f'Unsupported sequence shape: {sequence.shape}')

        frames = frames.float().clamp(0.0, 1.0)
        if frames.shape[1] == 1:
            frames = frames.repeat(1, 3, 1, 1)
        return frames

    def _extract_multiscale_features(self, frames):
        features = []
        current = frames
        for layer in self._feature_extractor:
            current = layer(current)
            features.append(current)
        return features

    def __call__(self, original, attacked):
        if original.shape != attacked.shape:
            raise ValueError(f'Perceptual loss requires matching shapes: {original.shape} vs {attacked.shape}')
        device = original.device
        self._feature_extractor = self._feature_extractor.to(device)
        self._feature_extractor.eval()

        original_frames = self._preprocess_sequence(original)
        attacked_frames = self._preprocess_sequence(attacked)

        original_features = self._extract_multiscale_features(original_frames.to(device))
        attacked_features = self._extract_multiscale_features(attacked_frames.to(device))

        losses = []
        for original_feat, attacked_feat in zip(original_features, attacked_features):
            if original_feat.shape != attacked_feat.shape:
                continue
            layer_loss = (original_feat - attacked_feat).abs().mean()
            losses.append(layer_loss)

        if not losses:
            return (original_frames - attacked_frames).abs().mean() * self.weight

        avg_loss = torch.stack(losses).mean() * self.weight
        return avg_loss * self.feature_scale

    def save_visualization(self, original, attacked, prefix='attack', limit=4):
        if not self.visualize or self.save_dir is None:
            return

        os.makedirs(self.save_dir, exist_ok=True)
        original_frames = original.detach().cpu().float().clamp(0.0, 1.0)
        attacked_frames = attacked.detach().cpu().float().clamp(0.0, 1.0)

        if original_frames.dim() == 5:
            sample_original = original_frames[0]
            sample_attacked = attacked_frames[0]
            if sample_original.shape[0] <= 4 and sample_original.shape[1] > 4:
                original_seq = sample_original.permute(1, 0, 2, 3)
                attacked_seq = sample_attacked.permute(1, 0, 2, 3)
            else:
                original_seq = sample_original.permute(1, 0, 2, 3)
                attacked_seq = sample_attacked.permute(1, 0, 2, 3)
        elif original_frames.dim() == 4:
            original_seq = original_frames[0].unsqueeze(1)
            attacked_seq = attacked_frames[0].unsqueeze(1)
        elif original_frames.dim() == 3:
            original_seq = original_frames.unsqueeze(1)
            attacked_seq = attacked_frames.unsqueeze(1)
        else:
            return

        if original_seq.shape[1] == 1:
            original_seq = original_seq.repeat(1, 3, 1, 1)
            attacked_seq = attacked_seq.repeat(1, 3, 1, 1)

        sample_count = min(original_seq.shape[0], attacked_seq.shape[0], limit)
        for frame_idx in range(sample_count):
            original_img = self._tensor_to_numpy(original_seq[frame_idx])
            attacked_img = self._tensor_to_numpy(attacked_seq[frame_idx])
            comparison = np.concatenate([original_img, attacked_img], axis=1)
            out_path = os.path.join(self.save_dir, f'{prefix}_{frame_idx:02d}.png')
            Image.fromarray(comparison).save(out_path)

    @staticmethod
    def _tensor_to_numpy(tensor):
        array = tensor.detach().cpu().numpy()
        if array.shape[0] == 1:
            array = array[0]
        elif array.shape[0] == 3:
            array = np.transpose(array, (1, 2, 0))
        else:
            array = np.transpose(array, (1, 2, 0))
        array = np.clip(array, 0.0, 1.0)
        array = np.uint8(array * 255.0)
        return array


class FrameSaliencyFGSMAttack:
    def __init__(self, epsilon=0.05, fraction_to_attack=0.2, clip_min=0.0, clip_max=1.0):
        """
        fraction_to_attack: The percentage of frames in the sequence to perturb (e.g., 0.20 = top 20%).
        """
        self.epsilon = epsilon
        self.fraction_to_attack = fraction_to_attack
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __call__(self, heatmaps, gradients):
        if gradients is None:
            return heatmaps

        # heatmaps shape: [Batch, Channels, Sequence, Height, Width]
        B, C, S, H, W = heatmaps.shape

        # 1. Calculate Frame Saliency Scores
        # Sum the absolute gradients over Channels, Height, and Width to get a score per frame
        frame_saliency = gradients.abs().sum(dim=(1, 3, 4))

        # 2. Determine exactly how many frames to attack based on the fraction
        k = max(1, int(S * self.fraction_to_attack))

        # 3. Find the indices of the Top-K most vulnerable frames
        _, topk_indices = torch.topk(frame_saliency, k, dim=1)

        # 4. Create a temporal mask to isolate those frames
        mask = torch.zeros((B, S), device=heatmaps.device)
        mask.scatter_(1, topk_indices, 1.0)

        # Reshape the mask to [B, 1, S, 1, 1] so it broadcasts across the spatial dimensions
        mask = mask.view(B, 1, S, 1, 1)

        # 5. Calculate the standard FGSM perturbation
        sign_data_grad = gradients.sign()
        perturbation = self.epsilon * sign_data_grad

        # 6. Apply the perturbation ONLY to the highly salient frames
        perturbed_heatmaps = heatmaps + (perturbation * mask)

        return torch.clamp(perturbed_heatmaps, self.clip_min, self.clip_max)

class FGSMSkeletonAttack:
    def __init__(self, epsilon=0.05, clip_min=0.0, clip_max=1.0):
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __call__(self, heatmaps, gradients):
        if gradients is None:
            return heatmaps

        # Calculate sign of loss gradient
        sign_data_grad = gradients.sign()

        # Perturb heatmaps
        perturbed_heatmaps = heatmaps + self.epsilon * sign_data_grad

        # Clip to valid range
        return torch.clamp(perturbed_heatmaps, self.clip_min, self.clip_max)


class EdgeSilhouetteAttack:
    def __init__(self, flip_probability=0.2):
        self.flip_prob = flip_probability
        self.edge_kernel = torch.tensor([[[[-1., -1., -1.],
                                           [-1., 8., -1.],
                                           [-1., -1., -1.]]]])

    def __call__(self, silhouettes):
        device = silhouettes.device
        B, C, S_dim, H, W = silhouettes.shape

        sils_2d = silhouettes.view(B * S_dim, 1, H, W).float()

        # Detect silhouette contours
        kernel = self.edge_kernel.to(device)
        edges = F.conv2d(sils_2d, kernel, padding=1)
        edges = (edges > 0).float()

        # Stochastic flip mask
        flip_mask = (torch.rand_like(sils_2d) < self.flip_prob).float()
        active_perturbation = edges * flip_mask

        # Invert edge pixels
        perturbed_sils_2d = torch.abs(sils_2d - active_perturbation)

        return perturbed_sils_2d.view(B, C, S_dim, H, W)
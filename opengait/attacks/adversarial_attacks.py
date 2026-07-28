import torch
import torch.nn as nn
import torch.nn.functional as F

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
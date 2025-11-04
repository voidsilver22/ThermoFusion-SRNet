# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure # <-- MODIFIED THIS LINE

# --- 1. CSRN Model Architecture ---

class ConvBlock(nn.Module):
    """Standard Convolutional Block: Conv -> BatchNorm -> ReLU"""
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class CrossAttentionBlock(nn.Module):
    """
    Cross-Attention Mechanism:
    - Query (Q) comes from the Thermal Stream (low-res features)
    - Key (K) and Value (V) come from the Optical Stream (high-res features)
    """
    def __init__(self, thermal_c, optical_c, embed_c):
        super().__init__()
        self.embed_c = embed_c
        
        # 1x1 convs to project features into Q, K, V space
        self.q_conv = nn.Conv2d(thermal_c, embed_c, 1)
        self.k_conv = nn.Conv2d(optical_c, embed_c, 1)
        self.v_conv = nn.Conv2d(optical_c, optical_c, 1) # V has full optical channels
        
        self.softmax = nn.Softmax(dim=-1)
        
        # Output projection
        self.out_conv = ConvBlock(optical_c, thermal_c, 1, 1, 0)
        
    def forward(self, x_thermal, x_optical):
        """
        x_thermal: [B, C_therm, H, W] (e.g., 64, 32, 32)
        x_optical: [B, C_opt, 3H, 3W] (e.g., 64, 32, 96)
        """
        # --- 1. Resample thermal features to match optical grid ---
        x_thermal_up = F.interpolate(
            x_thermal,
            size=x_optical.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        # --- 2. Generate Q, K, V ---
        B, _, H, W = x_optical.shape
        
        # Q from upsampled thermal
        q = self.q_conv(x_thermal_up).view(B, self.embed_c, H * W) # [B, C_embed, N]
        
        # K, V from optical
        k = self.k_conv(x_optical).view(B, self.embed_c, H * W) # [B, C_embed, N]
        v = self.v_conv(x_optical).view(B, -1, H * W) # [B, C_opt, N]
        
        # --- 3. Calculate Attention ---
        # Q.T * K
        attn = torch.bmm(q.transpose(1, 2), k) # [B, N, N]
        attn = self.softmax(attn / (self.embed_c ** 0.5)) # Scaled Dot-Product
        
        # Attn * V
        # (V.T * Attn.T).T
        out = torch.bmm(v, attn.transpose(1, 2)) # [B, C_opt, N]
        out = out.view(B, -1, H, W) # [B, C_opt, H, W]
        
        # --- 4. Project attended features back to thermal space ---
        # This is the "guidance" signal
        guidance = self.out_conv(out)
        
        # Return guidance to be added to the upsampled thermal stream
        return guidance


class CSRN(nn.Module):
    """
    Coupled Two-Stream Super-Resolution Network (CSRN)
    Upscale Factor: 3x
    """
    def __init__(self, in_c_lst=1, in_c_guide=8, base_c=64, embed_c=32):
        # --- NOTE: in_c_guide is now 8 (7 OLI + 1 Emissivity) ---
        super().__init__()
        
        # --- TIR Stream (Fidelity) ---
        self.tir_stream = nn.Sequential(
            ConvBlock(in_c_lst, base_c),   # [B, 64, H/3, W/3]
            ConvBlock(base_c, base_c),
            ConvBlock(base_c, base_c * 2)  # [B, 128, H/3, W/3]
        )
        
        # --- Optical Stream (Guidance) ---
        self.optical_stream = nn.Sequential(
            ConvBlock(in_c_guide, base_c), # [B, 64, H, W]
            ConvBlock(base_c, base_c),
            ConvBlock(base_c, base_c * 2)   # [B, 128, H, W]
        )
        
        # --- Cross-Attention Fusion ---
        self.fusion = CrossAttentionBlock(
            thermal_c=base_c * 2,
            optical_c=base_c * 2,
            embed_c=embed_c
        )
        
        # --- Upsampling & Output ---
        # --- Refinement & Output ---
        # We removed the PixelShuffle; upsampling is now handled
        # by F.interpolate in the forward pass.
        self.refinement = nn.Sequential(
        # Input is already [B, 128, H, W]
        ConvBlock(base_c * 2, base_c),
        ConvBlock(base_c, base_c),
        nn.Conv2d(base_c, 1, 1, 1, 0) # Output [B, 1, H, W]
        )

    def forward(self, x_lst, x_guide):
        """
        x_lst: Low-res LST input [B, 1, H/3, W/3]
        x_guide: High-res guide input [B, 8, H, W]
        """
        # 1. Run streams
        feat_tir = self.tir_stream(x_lst)     # [B, 128, H/3, W/3]
        feat_opt = self.optical_stream(x_guide) # [B, 128, H, W]
        
        # 2. Get guidance from fusion
        # Note: feat_tir is Q, feat_opt is K, V
        guidance = self.fusion(feat_tir, feat_opt) # [B, 128, H, W]
        
        # 3. Upsample TIR features
        # 3. Upsample TIR features
        # --- FIX: Interpolate to the *exact* size of the guidance tensor ---
        feat_tir_up = F.interpolate(
            feat_tir,
            size=guidance.shape[2:],  # Use guidance shape (88, 88)
            mode='bilinear',
            align_corners=False
        ) # [B, 128, H, W]
        
        # 4. Add guidance to upsampled features (Residual Connection)
        fused_features = feat_tir_up + guidance
        
        # 5. Generate final SR output
        sr_output = self.refinement(fused_features) # [B, 1, H, W]

        return sr_output

# --- 2. Physics-Informed Loss Function ---

class PhysicsInformedLoss(nn.Module):
    """
    Implements the combined loss:
    L_Total = lambda_K * L_K + lambda_SSIM * L_SSIM + lambda_Degrad * L_Degrad
    """
    def __init__(self, lambda_k=1.0, lambda_ssim=0.1, lambda_degrad=0.5, upscale_factor=3):
        super().__init__()
        self.lambda_k = lambda_k
        self.lambda_ssim = lambda_ssim
        self.lambda_degrad = lambda_degrad
        
        self.mse_loss = nn.MSELoss()
        
        # Note: SSIM needs a data range. Since we normalize to ~[-1, 1],
        # a data_range of 2.0-4.0 is reasonable.
        self.ssim_loss = StructuralSimilarityIndexMeasure(data_range=3.0)
        
        # The Degradation Operator D(.)
        # A 3x3 average pooling with stride 3
        self.degradation_op = nn.AvgPool2d(
            kernel_size=upscale_factor,
            stride= upscale_factor
        )
        
    def forward(self, pred_hr, target_hr, input_lr, device):
        """
        Args:
            pred_hr (torch.Tensor): Predicted HR LST (LST_30m_hat)
            target_hr (torch.Tensor): Target HR LST (LST_30m_Target)
            input_lr (torch.Tensor): Original LR LST (LST_100m_Aligned)
            device (torch.device): Device to run SSIM loss on
        """
        
        # --- Kinetic Loss (L_K) ---
        # MSE on the (normalized) Kelvin values
        loss_k = self.mse_loss(pred_hr, target_hr)
        
        # --- SSIM Loss (L_SSIM) ---
        # SSIM is minimized, so (1 - SSIM) is the loss.
        # Move SSIM metric to the correct device
        self.ssim_loss = self.ssim_loss.to(device)
        ssim_val = self.ssim_loss(pred_hr, target_hr)
        loss_ssim = 1.0 - ssim_val
        
        # --- Degradation Consistency Loss (L_Degrad) ---
        # Apply the degradation operator D(.) to the SR output
        pred_degraded = self.degradation_op(pred_hr)
        
        # Calculate MSE between degraded prediction and original LR input
        loss_degrad = self.mse_loss(pred_degraded, input_lr)
        
        # --- Total Loss ---
        total_loss = (self.lambda_k * loss_k) + \
                     (self.lambda_ssim * loss_ssim) + \
                     (self.lambda_degrad * loss_degrad)
                     
        # Return total loss and individual components for monitoring
        return total_loss, {
            "loss_k": loss_k.item(),
            "loss_ssim": loss_ssim.item(),
            "loss_degrad": loss_degrad.item()
        }
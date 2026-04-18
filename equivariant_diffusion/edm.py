from __future__ import annotations
import math
from typing import Dict

import numpy as np 
import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean

import utils
from torch import Tensor


class Diffusion(nn.Module):


    def __init__(self, dynamics: nn.Module, atom_nf: int, residue_nf: int,
                 n_dims: int, size_histogram: Dict,
                 # EDM/ Parameters / Position
                 sigma_min_pos: float = 0.0004, sigma_max_pos: float = 80.0,
                 sigma_data_pos: float = 2.5, P_mean_pos: float = -1.2, P_std_pos: float = 1.5,
                 # EDM/ Parameters / Atom Types
                 sigma_min_feat: float = 0.0004, sigma_max_feat: float = 80.0,
                 sigma_data_feat: float = 0.6, P_mean_feat: float = -1.2, P_std_feat: float = 1.5,

                 rho: float = 7,
                 dilated_schedule: bool = False,
                 tau_start: float = 0.6,
                 tau_end: float = 0.8,            
                 #Sampling defaults
                 num_sampling_steps: int = 40,
                 norm_values=(1., 1.), norm_biases=(None, 0.),
                 virtual_node_idx=None,
                 scale_pocket_coords: bool = False,
                 atom_type_prior=None,
                 loss_x_weight: float = 10.0, 
                 ):
        super().__init__()

        self.loss_x_weight = loss_x_weight
        self.scale_pocket_coords = scale_pocket_coords
        self.dynamics = dynamics
        self.atom_nf = atom_nf
        self.residue_nf = residue_nf
        self.n_dims = n_dims
        self.num_classes = self.atom_nf

        # EDM Parameters
        self.sigma_min_pos = sigma_min_pos
        self.sigma_max_pos = sigma_max_pos
        self.sigma_data_pos = sigma_data_pos
        self.P_mean_pos = P_mean_pos
        self.P_std_pos = P_std_pos

        self.sigma_min_feat = sigma_min_feat
        self.sigma_max_feat = sigma_max_feat
        self.sigma_data_feat = sigma_data_feat
        self.P_mean_feat = P_mean_feat
        self.P_std_feat = P_std_feat

        self.rho = rho
        self.num_sampling_steps = num_sampling_steps
        self.norm_values = norm_values
        self.norm_biases = norm_biases
        
        #Dilated
        self.dilated_schedule = dilated_schedule
        self.tau_start = tau_start
        self.tau_end = tau_end
        
        # Distribution of nodes (giữ lại để sampling số lượng node nếu cần)
        self.size_distribution = DistributionNodes(size_histogram)
        self.vnode_idx = virtual_node_idx

        if atom_type_prior is not None:
            assert atom_type_prior.shape[0] == atom_nf, \
                f"Prior dim {atom_type_prior.shape[0]} != atom_nf {atom_nf}"
            log_prior = torch.log(atom_type_prior.float() + 1e-10)
            self.register_buffer('log_atom_prior', log_prior)
        else:
            # Uniform prior fallback (equivalent to no prior)
            self.register_buffer('log_atom_prior', torch.zeros(atom_nf))

    @property
    def device(self):
        return next(self.parameters()).device
    
    #EDM PRECONDITIONING

    def c_skip(self,sigma, sigma_data):
        return (sigma_data ** 2) / (sigma ** 2 + sigma_data ** 2)
    
    def c_out(self, sigma, sigma_data):
        return sigma * sigma_data / (sigma ** 2 + sigma_data ** 2).sqrt()
    
    def c_in(self, sigma, sigma_data):
        return 1 / (sigma_data ** 2 + sigma ** 2).sqrt()
    
    def c_noise(self, sigma, sigma_data): 
        return 0.25 * (sigma / sigma_data).log() 
    
    def loss_weight(self, sigma, sigma_data):
        return (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
    
    def noise_distribution_pos(self, batch_size):
        """Sample noise cho coordinates"""
        rnd = torch.randn((batch_size,), device=self.device)
        sigma = self.sigma_data_pos * (self.P_mean_pos + self.P_std_pos * rnd).exp()
        return sigma.clamp(self.sigma_min_pos, self.sigma_max_pos)
    
    def noise_distribution_feat(self, batch_size):
        """Sample noise cho atom features"""
        rnd = torch.randn((batch_size,), device=self.device)
        sigma = self.sigma_data_feat * (self.P_mean_feat + self.P_std_feat * rnd).exp()
        return sigma.clamp(self.sigma_min_feat, self.sigma_max_feat)
    
    @staticmethod
    def inflate_batch_array(array, target):
        """
        Inflates the batch array (array) with only a single axis
        (i.e. shape = (batch_size,), or possibly more emty axes
        (i.e. shape (batch_sizem 1,...,1)) to match the target shape
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size())-1)
        return array.view(target_shape)
    
    def normalize(self, ligand = None, pocket = None):
        if ligand is not None:
            ligand['x'] = ligand['x'] / self.norm_values[0]
            raw_h = ligand['one_hot'].float() * 2.0 - 1.0

            ligand['one_hot'] = (raw_h - self.norm_biases[1]) / self.norm_values[1]
        
        if pocket is not None:
            pocket['x'] = pocket['x'] / self.norm_values[0]
            raw_h_pocket = pocket['one_hot'].float() * 2.0 - 1.0
            pocket['one_hot'] = (raw_h_pocket - self.norm_biases[1]) / self.norm_values[1]
        
        return ligand, pocket
    
    def unnormalize(self, x, h_cat):
        x = x*self.norm_values[0]
        h_cat = h_cat * self.norm_values[1] + self.norm_biases[1]

        return x, h_cat
    
    def unnormalize_z(self, z_lig, z_pocket):
        
        x_lig, h_lig = z_lig[:, :self.n_dims], z_lig[:, self.n_dims:]
        x_pocket, h_pocket = z_pocket[:, :self.n_dims], z_pocket[:, self.n_dims:]

        # Unnormalize
        x_lig, h_lig = self.unnormalize(x_lig, h_lig)
        x_pocket, h_pocket = self.unnormalize(x_pocket, h_pocket)

        return torch.cat([x_lig, h_lig], dim = 1), \
               torch.cat([x_pocket, h_pocket], dim = 1)
    
    @staticmethod
    def remove_mean_batch(x_lig, x_pocket, lig_indicies, pocket_indicies):
        mean = scatter_mean(x_lig, lig_indicies, dim=0)

        x_lig = x_lig - mean[lig_indicies]
        x_pocket = x_pocket - mean[pocket_indicies]
        return x_lig, x_pocket
    
    @staticmethod
    def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
        largest_value = x.abs().max().item()
        error = scatter_add(x, node_mask, dim = 0).abs().max().item()
        rel_error = error / (largest_value + eps)
        assert rel_error < 1e-2, f'Mean is not zero, relative_error {rel_error}'

    
    @staticmethod
    def sum_except_batch(x, indices):
        return scatter_add(x.sum(-1), indices, dim=0)
   

    def compute_type_posterior(self, h_noised, sigma_h, normalize_to_prob_space=True):
        """
        Compute exact Bayes posterior p(z=k | h_t, sigma_h) for atom type classification.
        
        The posterior follows from applying Bayes' rule to Gaussian likelihood under the 
        VE noising process: h_t = h_0 + sigma_h * epsilon, where h_0 ∈ {-1, +1}^K.
        
        Args:
            h_noised: [N_atoms, K] — Noised features in {-1,+1} + noise space 
                    (BEFORE c_in preconditioning)
            sigma_h: [N_atoms, 1] — Per-atom noise level sigma_h
            normalize_to_prob_space: bool — If True, output in [0,1] (probability simplex).
                                            If False, output raw logits.
        
        Returns:
            posterior: [N_atoms, K] — Posterior probability distribution over atom types
                    (lives on probability simplex: non-negative, sums to 1)
        """
        # sigma_h has shape [N_atoms, 1], h_noised has shape [N_atoms, K]
        # Clamp sigma to avoid division by zero at the end of sampling
        sigma_sq = torch.clamp(sigma_h ** 2, min=1e-6)  # [N_atoms, 1]
        
        # Core formula: logits = 2 * h_t / sigma_h^2 + log(prior)
        # The 2/sigma^2 factor comes from the Gaussian log-likelihood difference
        logits = 2.0 * h_noised / sigma_sq + self.log_atom_prior.unsqueeze(0)
        # logits shape: [N_atoms, K]
        
        # Numerical stability: clamp logits to prevent overflow in softmax
        # At sigma_h = 0.01, 2/sigma^2 = 20000, but h_noised ≈ ±1 so logits ≈ ±20000
        # Softmax only cares about relative differences, so clamping is safe
        logits = torch.clamp(logits, min=-88.0, max=88.0)  # exp(88) ≈ 1.6e38, safe for float32
        
        if normalize_to_prob_space:
            return F.softmax(logits, dim=-1)
        else:
            return logits
# ==================== FORWARD (TRAINING) ====================
    def forward(self, ligand, pocket, return_info=False):

        # 1. Normalize & Prepare Data
        ligand, pocket = self.normalize(ligand, pocket)
        
        x_lig = ligand['x']
        h_lig = ligand['one_hot']
        
        # Center Coordinates 
        x_lig, x_pocket = self.remove_mean_batch(
            x_lig, pocket['x'], ligand['mask'], pocket['mask']
        )

        batch_size = len(ligand['size'])

        # 2. Sample Noise Độc lập
        # X: Position Noise
        sigma_x = self.noise_distribution_pos(batch_size)
        sigma_x_mapped = sigma_x[ligand['mask']].unsqueeze(1)

        # H: Feature Noise
        sigma_h = self.noise_distribution_feat(batch_size)
        sigma_h_mapped = sigma_h[ligand['mask']].unsqueeze(1)

        # 3. Add Noise
        noise_x = torch.randn_like(x_lig)
        noise_h = torch.randn_like(h_lig)
        
        x_noised = x_lig + sigma_x_mapped * noise_x
        h_noised = h_lig + sigma_h_mapped * noise_h

        # 4. Preconditioning coefficients (Tách biệt)
        # --- Cho Tọa độ ---
        cin_x = self.c_in(sigma_x_mapped, self.sigma_data_pos)
        cout_x = self.c_out(sigma_x_mapped, self.sigma_data_pos)
        cskip_x = self.c_skip(sigma_x_mapped, self.sigma_data_pos)

        cin_x_batch = self.c_in(sigma_x, self.sigma_data_pos)
        inv_scale = 1.0 / cin_x_batch

        #Node_attr
        lig_type_posterior = self.compute_type_posterior(h_noised, sigma_h_mapped)

        # --- Cho Features ---
        cin_h = self.c_in(sigma_h_mapped, self.sigma_data_feat)
        cout_h = self.c_out(sigma_h_mapped, self.sigma_data_feat)
        cskip_h = self.c_skip(sigma_h_mapped, self.sigma_data_feat)
        
        # 5. Time Embedding (Dual Input)
        # Tạo vector [Batch, 2] chứa thông tin noise của cả X và H
        c_noise_x = (sigma_x / self.sigma_data_pos).log() * 0.25
        c_noise_h = (sigma_h / self.sigma_data_feat).log() * 0.25 
        c_noise = torch.stack([c_noise_x, c_noise_h], dim=-1)


        # 6. Network Input
        net_input = torch.cat([x_noised * cin_x, h_noised * cin_h], dim=-1)

        if self.scale_pocket_coords:
            # Mode 1: Scale Pocket
            sigma_x_pocket_mapped = sigma_x[pocket['mask']].unsqueeze(1)
            cin_x_pocket = self.c_in(sigma_x_pocket_mapped, self.sigma_data_pos)
            x_pocket_input = x_pocket * cin_x_pocket
        else:
            # Mode 2: No Scale Pocket 
            x_pocket_input = x_pocket

        xh_pocket = torch.cat([x_pocket_input, pocket['one_hot']], dim=1)

        # 7. Dynamics Call
        raw_update, _ = self.dynamics(
            net_input, xh_pocket, c_noise, 
            ligand['mask'], pocket['mask'],
            current_scale=inv_scale,
            lig_type_posterior=lig_type_posterior
        )
        
        update_x = raw_update[:, :self.n_dims]
        update_h = raw_update[:, self.n_dims:]

        # 8. Denoised Estimate
        denoised_x = cskip_x * x_noised + cout_x * update_x
        denoised_h = cskip_h * h_noised + cout_h * update_h

        # 9. Loss Calculation
        # --- Loss X ---
            
        coord_diff_sq = ((denoised_x - x_lig) ** 2) 
        loss_x_per_atom = coord_diff_sq.sum(dim=-1)
        mse_loss_per_graph = scatter_add(loss_x_per_atom, ligand['mask'], dim=0) 
        div_factor = 3.0 * ligand['size'].float().to(self.device)
        mse_loss_per_graph = mse_loss_per_graph / (div_factor + 1e-5)
        
        w_x = self.loss_weight(sigma_x, self.sigma_data_pos)
        loss_x = (mse_loss_per_graph * w_x).mean()

        #Debug
        edge_stats = {}
        with torch.no_grad():
            sig = sigma_x
            mask_high = sig > 10.0
            mask_mid = (sig <= 10.0) & (sig > 0.5)
            mask_low = sig <= 0.5
            
            # Tính loss riêng cho từng vùng (nếu vùng đó có dữ liệu trong batch)
            loss_high = (mse_loss_per_graph[mask_high] * w_x[mask_high]).mean() if mask_high.any() else 0.0
            loss_mid = (mse_loss_per_graph[mask_mid] * w_x[mask_mid]).mean() if mask_mid.any() else 0.0
            loss_low = (mse_loss_per_graph[mask_low] * w_x[mask_low]).mean() if mask_low.any() else 0.0

        #Debug--

        # --- Loss H ---
        w_h = self.loss_weight(sigma_h, self.sigma_data_feat)
        loss_h_sq = (denoised_h - h_lig) ** 2
        loss_h_sq_sum = scatter_add(loss_h_sq.sum(dim=-1), ligand['mask'], dim=0)
        loss_h = (w_h * loss_h_sq_sum / (ligand['size'].float() * self.atom_nf + 1e-5)).mean()

        loss = self.loss_x_weight * loss_x + loss_h
        
        xh_lig_hat = torch.cat([denoised_x, denoised_h], dim=-1)

        info = {
            'loss': loss.item(), 
            'loss_x': loss_x.item(),
            'loss_h': loss_h.item(),
            'sigma_x': sigma_x.mean().item(),
            'sigma_h': sigma_h.mean().item(),
            'xh_lig_hat': xh_lig_hat,
            'debug/loss_x_high_noise': loss_high.item() if isinstance(loss_high, torch.Tensor) else loss_high,
            'debug/loss_x_mid_noise': loss_mid.item() if isinstance(loss_mid, torch.Tensor) else loss_mid,
            'debug/loss_x_low_noise': loss_low.item() if isinstance(loss_low, torch.Tensor) else loss_low,
        }
        info.update(edge_stats)

        return loss, info

    def get_sigma_schedule(self, num_sampling_steps=None, sigma_min=None, sigma_max=None, rho=None, dilated=None,
                       tau_start=None, tau_end=None):
        
        if num_sampling_steps is None:
            num_sampling_steps = self.num_sampling_steps
        if sigma_min is None:
            sigma_min = self.sigma_min_pos
        if sigma_max is None:
            sigma_max = self.sigma_max_pos

        if rho is None:
            rho = self.rho
        if dilated is None:
            dilated = getattr(self, 'dilated_schedule', False)

        device = self.device
        dtype = torch.float32
        
        steps = torch.arange(num_sampling_steps, device=device, dtype=dtype)
        tau = steps / (num_sampling_steps - 1) # Linear 0 -> 1


        if getattr(self, 'dilated_schedule', False):
            tau=self._apply_dilation(tau, tau_start=tau_start, tau_end=tau_end)   
        
        inv_rho = 1 / self.rho
        sigmas = (
            sigma_max ** inv_rho
            + tau * (sigma_min**inv_rho - sigma_max**inv_rho)
        ) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value=0.0)
        
        return sigmas
    
    def get_sigma_schedules(self, num_sampling_steps=None, sigma_min_pos=None, sigma_max_pos=None,
                        sigma_min_feat=None, sigma_max_feat=None,
                        rho=None, dilated=None,
                        tau_start=None, tau_end=None):

        sigmas_pos = self.get_sigma_schedule(
        num_sampling_steps,
        sigma_min=sigma_min_pos if sigma_min_pos is not None else self.sigma_min_pos,
        sigma_max=sigma_max_pos if sigma_max_pos is not None else self.sigma_max_pos,
        rho=rho,
        dilated=dilated,
        tau_start=tau_start,
        tau_end=tau_end,
        )
        sigmas_feat = self.get_sigma_schedule(
            num_sampling_steps,
            sigma_min=sigma_min_feat if sigma_min_feat is not None else self.sigma_min_feat,
            sigma_max=sigma_max_feat if sigma_max_feat is not None else self.sigma_max_feat,
            rho=rho,
            dilated=dilated,
            tau_start=tau_start,
            tau_end=tau_end,
        )
        return sigmas_pos, sigmas_feat
    
    def _apply_dilation(self, tau, tau_start=None, tau_end=None):

        if tau_start is None:
            tau_start = self.tau_start
        if tau_end is None:
            tau_end = self.tau_end

        dilation  = 8.0 / 3.0

        x_interval = tau_end - tau_start
        r = (1.0 - dilation * x_interval) / (1.0 - x_interval)
        l = r * tau_start
        u = l + dilation * x_interval

        tau_d = torch.zeros_like(tau)
        mask_low  = tau < l
        mask_mid  = (tau >= l) & (tau < u)
        mask_high = tau >= u

        tau_d[mask_low]  = tau[mask_low] / r
        tau_d[mask_mid]  = (tau[mask_mid] - l) / dilation + tau_start
        tau_d[mask_high] = (tau[mask_high] - u) / r + tau_end
        return tau_d
    
    def sample_normal_zero_com(self, lig_mask, pocket_mask, xh0_pocket, sigma_x, sigma_h, mu_x=None, mu_h=None):

        eps_x = torch.randn(lig_mask.shape[0], self.n_dims, device=self.device)
        eps_h = torch.randn(lig_mask.shape[0], self.atom_nf, device=self.device)

        if mu_x is None:
            z_x = eps_x * sigma_x
            z_h = eps_h * sigma_h

        else:
            z_x = mu_x + eps_x * sigma_x
            z_h = mu_h + eps_h * sigma_h

        xh_pocket = xh0_pocket.clone()
        
        z_x, xh_pocket[:, :self.n_dims] = self.remove_mean_batch(
                    z_x, 
                    xh0_pocket[:, :self.n_dims], 
                    lig_mask, 
                    pocket_mask
                )
        z_xh_lig = torch.cat([z_x, z_h], dim=1)
        return z_xh_lig, xh_pocket
    
    def _denoise_step(self, z_x, z_h, x_pocket, h_pocket,
                  sigma_x_scalar, sigma_h_scalar,
                  lig_mask, pocket, batch_size, device):

        s_vec_x = torch.full((batch_size,), sigma_x_scalar, device=device)
        s_vec_h = torch.full((batch_size,), sigma_h_scalar, device=device)

        s_mapped_x = s_vec_x[lig_mask].unsqueeze(1)  
        s_mapped_h = s_vec_h[lig_mask].unsqueeze(1)

        lig_type_posterior = self.compute_type_posterior(z_h, s_mapped_h)

        if self.scale_pocket_coords:
            sigma_x_pocket = s_vec_x[pocket['mask']].unsqueeze(1)
            cin_x_pocket = self.c_in(sigma_x_pocket, self.sigma_data_pos)
            x_pocket_in = x_pocket * cin_x_pocket
        else:
            x_pocket_in = x_pocket
        xh_pocket = torch.cat([x_pocket_in, h_pocket], dim=1)

        c_noise_x = (s_vec_x / self.sigma_data_pos).log()  * 0.25
        c_noise_h = (s_vec_h / self.sigma_data_feat).log() * 0.25
        c_noise   = torch.stack([c_noise_x, c_noise_h], dim=-1)  

        cin_x  = self.c_in(s_mapped_x,  self.sigma_data_pos)
        cout_x = self.c_out(s_mapped_x, self.sigma_data_pos)
        cskip_x = self.c_skip(s_mapped_x, self.sigma_data_pos)

        cin_h  = self.c_in(s_mapped_h,  self.sigma_data_feat)
        cout_h = self.c_out(s_mapped_h, self.sigma_data_feat)
        cskip_h = self.c_skip(s_mapped_h, self.sigma_data_feat)

        cin_x_batch = self.c_in(s_vec_x, self.sigma_data_pos)
        inv_scale = 1.0 / cin_x_batch              

        net_in = torch.cat([z_x * cin_x, z_h * cin_h], dim=-1)

        raw_out, _ = self.dynamics(
            net_in, xh_pocket, c_noise,
            lig_mask, pocket['mask'],
            current_scale=inv_scale,
            lig_type_posterior=lig_type_posterior,
        )
        out_x, out_h = raw_out[:, :3], raw_out[:, 3:]

        d_x = cskip_x * z_x + cout_x * out_x
        d_h = cskip_h * z_h + cout_h * out_h

        grad_x = (z_x - d_x) / s_mapped_x
        grad_h = (z_h - d_h) / s_mapped_h

        return d_x, d_h, grad_x, grad_h
    
    def _diffforce_linear_scale(
        self,
        step_idx: int,
        n_steps: int,
        force_scale: float,
        force_start: float,
    ) -> float:

        active_start = int((1.0 - force_start) * n_steps)
        if step_idx < active_start:
            return 0.0
        ramp_len = max(n_steps - 1 - active_start, 1)
        fraction = (step_idx - active_start) / ramp_len
        return fraction * force_scale

    @staticmethod
    def _normalize_gradient(grad: torch.Tensor) -> torch.Tensor:
        """
        Per-atom L2 normalisation của ∇U (Appendix B của paper).

        Đảm bảo λ_sc có ý nghĩa nhất quán với mọi kích thước phân tử.
        Atoms không có interaction (||∇U_i|| ≈ 0) → correction = 0.
        """
        norms = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return grad / norms
    
    @torch.no_grad()
    def sample_denovo(self, pocket, num_nodes_lig, return_frames=1, num_sampling_steps=None,
                      sigma_min_pos=None, sigma_max_pos=None,
                      sigma_min_feat=None, sigma_max_feat=None,
                      rho=None, dilated=None,
                      tau_start=None, tau_end=None,
                      ):
        device = self.device

        n_steps = num_sampling_steps if num_sampling_steps is not None \
              else self.num_sampling_steps

        batch_size = pocket['size'].shape[0]
        pocket_original_com = scatter_mean(pocket['x'], pocket['mask'], dim=0) 

        _, pocket = self.normalize(pocket=pocket)
        xh_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)

        # 2. Prepare Mask & Noise Schedule
        lig_mask = utils.num_nodes_to_batch_mask(batch_size, num_nodes_lig, device)
        
        sigmas_x_seq, sigmas_h_seq = self.get_sigma_schedules(
            n_steps,
            sigma_min_pos=sigma_min_pos,
            sigma_max_pos=sigma_max_pos,
            sigma_min_feat=sigma_min_feat,
            sigma_max_feat=sigma_max_feat,
            rho=rho,
            dilated=dilated,
            tau_start=tau_start,
            tau_end=tau_end,
        )

        z_xh_ligand, xh_pocket = self.sample_normal_zero_com(
                    lig_mask=lig_mask,
                    pocket_mask=pocket['mask'],
                    xh0_pocket=xh_pocket,
                    sigma_x=sigmas_x_seq[0], 
                    sigma_h=sigmas_h_seq[0]
                )
        
        x_pocket = xh_pocket[:, :self.n_dims]
        h_pocket = xh_pocket[:, self.n_dims:]
        
        z_x = z_xh_ligand[:, :self.n_dims]
        z_h = z_xh_ligand[:, self.n_dims:]

        out_x_chain = []
        out_h_chain = []
        save_indices = set(torch.linspace(0, n_steps - 1, return_frames).round().long().tolist())
        
        batch_idx_lig = utils.num_nodes_to_batch_mask(batch_size, num_nodes_lig, device)
        offset_vector = pocket_original_com[batch_idx_lig]

        for i in range(n_steps):

            if i in save_indices:
                x_curr, h_curr = self.unnormalize(z_x, z_h)
                x_curr = x_curr + offset_vector
                out_x_chain.append(x_curr.cpu())
                out_h_chain.append(h_curr.cpu())

            # Lấy sigma hiện tại và kế tiếp
            sigma_x_cur = sigmas_x_seq[i]
            sigma_h_cur = sigmas_h_seq[i]
            
            sigma_x_next = sigmas_x_seq[i + 1]
            sigma_h_next = sigmas_h_seq[i + 1]

            _, _, grad_x, grad_h = self._denoise_step(
            z_x, z_h, x_pocket, h_pocket,
            sigma_x_cur, sigma_h_cur,
            lig_mask, pocket, batch_size, device,
            )

            dt_x = sigma_x_next - sigma_x_cur
            dt_h = sigma_h_next - sigma_h_cur

            next_x = z_x + grad_x * dt_x
            next_h = z_h + grad_h * dt_h

            # --- 2. Correction (Heun) ---
            if sigma_x_next > 0 and sigma_h_next >0:

                _, _, grad_x_2, grad_h_2 = self._denoise_step(
                next_x, next_h, x_pocket, h_pocket,
                sigma_x_next, sigma_h_next,
                lig_mask, pocket, batch_size, device,
                )       
                
                next_x = z_x + (grad_x + grad_x_2) * 0.5 * dt_x
                next_h = z_h + (grad_h + grad_h_2) * 0.5 * dt_h

            z_x = next_x
            z_h = next_h
            
            z_x, x_pocket  = self.remove_mean_batch(
                            z_x, 
                            x_pocket, 
                            lig_mask, 
                            pocket['mask']
                        )

        x_final, h_final_unnorm = self.unnormalize(z_x, z_h)
        
        x_final = x_final + offset_vector

        h_final_idx = torch.argmax(h_final_unnorm, dim=-1)
        h_final = F.one_hot(h_final_idx, num_classes=self.atom_nf).float()

        if return_frames > 1:
            out_x_chain.append(x_final.cpu())
            _, h_final_cont = self.unnormalize(z_x, z_h)
            out_h_chain.append(h_final_cont.cpu())
            return torch.stack(out_x_chain), torch.stack(out_h_chain)
        else:
            return x_final, h_final
        
    @staticmethod
    def _build_repaint_schedule(n_steps: int, jump_length: int):

        if jump_length <= 1:
            return [('denoise', i) for i in range(n_steps)]
    
        schedule = []
        chunk_start = 0
    
        while chunk_start < n_steps:
            chunk_end = min(chunk_start + jump_length, n_steps)
            is_last_chunk = (chunk_end >= n_steps)
    
            # First forward pass through this chunk
            for s in range(chunk_start, chunk_end):
                schedule.append(('denoise', s))
    
            if not is_last_chunk:
                # Backward jump: renoise from σ[chunk_end] back to σ[chunk_start]
                schedule.append(('renoise', chunk_start, chunk_end))
    
                # Second forward pass (redo the chunk after jump-back)
                for s in range(chunk_start, chunk_end):
                    schedule.append(('denoise', s))
    
            chunk_start = chunk_end
    
        return schedule
    
    
    # ──────────────────────────────────────────────────────────────────────────
    # Main inpaint method
    # ──────────────────────────────────────────────────────────────────────────
    
    @torch.no_grad()
    def inpaint(
        self,
        ligand: dict,
        pocket: dict,
        lig_mask_fixed: torch.Tensor,
        *,
        resamplings: int = 1,
        jump_length: int = 1,
        timesteps: int = None,
        return_frames: int = 1,
        sigma_min_pos: float = None,
        sigma_max_pos: float = None,
        sigma_min_feat: float = None,
        sigma_max_feat: float = None,
        rho: float = None,
        dilated: bool = None,
        tau_start: float = None,
        tau_end: float = None,
        repulsion_scale: float = 0.0,
        repulsion_cutoff: float = 1.2,
        repulsion_start_frac: float = 0.5,
    ):
        """
        Repaint inpainting — v2 with pre-computed schedule.
    
        Key parameters
        ─────────────
        resamplings   : r — per-step resampling iterations (inner loop of RePaint)
        jump_length   : j — chunk size for backward jumps (j=1 → no jumps)
                        Paper final settings: j=10, r=10
    
        The schedule is pre-computed as a finite list of actions,
        making infinite loops structurally impossible.
        """
        device = self.device
        n_steps = timesteps if timesteps is not None else self.num_sampling_steps
        batch_size = len(ligand['size'])
    
        # ── 0. Save originals ────────────────────────────────────────────
        x_lig_original = ligand['x'].clone()
        h_lig_original = ligand['one_hot'].clone()
        inpaint_mask = lig_mask_fixed.bool()
    
        # ── 1. Normalize ─────────────────────────────────────────────────
        ligand, pocket = self.normalize(ligand, pocket)
    
        x_lig_0 = ligand['x'].clone()
        h_lig_0 = ligand['one_hot'].clone()
        lig_mask = ligand['mask']
        pocket_mask = pocket['mask']
        pocket_com_norm = scatter_mean(pocket['x'], pocket_mask, dim=0,
                                       dim_size=batch_size)
    
        # ── 2. Centering from fixed atoms ────────────────────────────────
        if inpaint_mask.any():
            n_fixed = scatter_add(inpaint_mask.float(), lig_mask, dim=0,
                                dim_size=batch_size)
            x_fix_only = torch.zeros_like(x_lig_0)
            x_fix_only[inpaint_mask] = x_lig_0[inpaint_mask]
            sum_fixed = scatter_add(x_fix_only, lig_mask, dim=0,
                                    dim_size=batch_size)
            com_fixed = sum_fixed / n_fixed.unsqueeze(-1).clamp(min=1)
    
            has_fixed = n_fixed > 0
            if not has_fixed.all():
                com_all = scatter_mean(x_lig_0, lig_mask, dim=0,
                                    dim_size=batch_size)
                com_fixed[~has_fixed] = pocket_com_norm[~has_fixed]
        else:
            com_fixed = pocket_com_norm
    
        centering_com_orig = com_fixed * self.norm_values[0]
    
        x_lig_0 = x_lig_0 - com_fixed[lig_mask]
        pocket_x = pocket['x'] - com_fixed[pocket_mask]
    
        # Immutable references
        x_lig_0_frozen = x_lig_0.clone()
        h_lig_0_frozen = h_lig_0.clone()
        frame_offset = torch.zeros(batch_size, 3, device=device,
                                dtype=x_lig_0.dtype)
        h_pocket = pocket['one_hot']
    
        # ── 3. Sigma schedules ───────────────────────────────────────────
        sigmas_x_seq, sigmas_h_seq = self.get_sigma_schedules(
            n_steps,
            sigma_min_pos=sigma_min_pos, sigma_max_pos=sigma_max_pos,
            sigma_min_feat=sigma_min_feat, sigma_max_feat=sigma_max_feat,
            rho=rho, dilated=dilated,
            tau_start=tau_start, tau_end=tau_end,
        )
    
        # ── 4. Initialize at σ_max ───────────────────────────────────────
        n_lig_atoms = lig_mask.shape[0]
        eps_x = torch.randn(n_lig_atoms, self.n_dims, device=device)
        eps_h = torch.randn(n_lig_atoms, self.atom_nf, device=device)
    
        sigma_x_init = sigmas_x_seq[0]
        sigma_h_init = sigmas_h_seq[0]
    
        z_x = eps_x * sigma_x_init
        z_h = eps_h * sigma_h_init
    
        # FIX 5: Fixed atoms from forward-noised GT
        if inpaint_mask.any():
            z_x[inpaint_mask] = (x_lig_0_frozen[inpaint_mask]
                                + sigma_x_init * torch.randn_like(
                                    x_lig_0_frozen[inpaint_mask]))
            z_h[inpaint_mask] = (h_lig_0_frozen[inpaint_mask]
                                + sigma_h_init * torch.randn_like(
                                    h_lig_0_frozen[inpaint_mask]))
    
        com_init = scatter_mean(z_x, lig_mask, dim=0)
        z_x = z_x - com_init[lig_mask]
        x_pocket = pocket_x - com_init[pocket_mask]
        frame_offset = frame_offset + com_init
    
        pocket_ctx = {'mask': pocket_mask, 'size': pocket['size']}
    
        # ── 5. Build pre-computed schedule ───────────────────────────────
        schedule = self._build_repaint_schedule(n_steps, jump_length)
    
        # Track which "sigma level" our signal is at.
        # After denoise step i, the signal is at sigma[i+1].
        # This is needed to compute the correct renoise variance.
        current_sigma_idx = 0  # initially at sigma[0] (= σ_max)
    
        # ── 6. Execute schedule ──────────────────────────────────────────
        for action in schedule:
    
            if action[0] == 'renoise':
                # ── Backward jump ────────────────────────────────────
                target_idx, source_idx = action[1], action[2]
    
                # Signal is currently at σ[source_idx].
                # Add noise to bring it back to σ[target_idx] (higher sigma).
                sigma_x_target = sigmas_x_seq[target_idx]
                sigma_h_target = sigmas_h_seq[target_idx]
                sigma_x_source = sigmas_x_seq[source_idx]
                sigma_h_source = sigmas_h_seq[source_idx]
    
                # Variance to add: σ_target² − σ_source²
                # (σ_target > σ_source since target is earlier = higher noise)
                std_x = (sigma_x_target**2 - sigma_x_source**2).clamp(min=0.).sqrt()
                std_h = (sigma_h_target**2 - sigma_h_source**2).clamp(min=0.).sqrt()
    
                if std_x > 1e-8:
                    z_x = z_x + std_x * torch.randn_like(z_x)
                    z_h = z_h + std_h * torch.randn_like(z_h)
                    if inpaint_mask.any():
                        x_lig_0_current = x_lig_0_frozen - frame_offset[lig_mask]
 
                        z_x[inpaint_mask] = (
                            x_lig_0_current[inpaint_mask]
                            + sigma_x_target
                            * torch.randn_like(x_lig_0_current[inpaint_mask])
                        )
                        z_h[inpaint_mask] = (
                            h_lig_0_frozen[inpaint_mask]
                            + sigma_h_target
                            * torch.randn_like(h_lig_0_frozen[inpaint_mask])
                        )
                    # Re-centre (FIX 3: fixed-atom CoM)
                    com_j = self._fixed_atom_com(
                        z_x, lig_mask, inpaint_mask, batch_size
                    )
                    z_x      = z_x      - com_j[lig_mask]
                    x_pocket = x_pocket - com_j[pocket_mask]
                    frame_offset = frame_offset + com_j
    
                current_sigma_idx = target_idx
                continue
    
            # ── Forward denoise step ─────────────────────────────────
            assert action[0] == 'denoise'
            step_i = action[1]
    
            sigma_x_cur = sigmas_x_seq[step_i]
            sigma_h_cur = sigmas_h_seq[step_i]
            sigma_x_nxt = sigmas_x_seq[step_i + 1]
            sigma_h_nxt = sigmas_h_seq[step_i + 1]
    
            dt_x = sigma_x_nxt - sigma_x_cur
            dt_h = sigma_h_nxt - sigma_h_cur
    
            # ── Repaint inner loop (r resamplings per step) ──────────
            for k in range(1, resamplings + 1):
    
                # A. Euler denoise
                _, _, grad_x, grad_h = self._denoise_step(
                    z_x, z_h, x_pocket, h_pocket,
                    sigma_x_cur, sigma_h_cur,
                    lig_mask, pocket_ctx, batch_size, device,
                )
                x_prev_gen = z_x + grad_x * dt_x
                h_prev_gen = z_h + grad_h * dt_h
    
                # Heun correction
                if sigma_x_nxt > 0 and sigma_h_nxt > 0:
                    _, _, grad_x_2, grad_h_2 = self._denoise_step(
                        x_prev_gen, h_prev_gen, x_pocket, h_pocket,
                        sigma_x_nxt, sigma_h_nxt,
                        lig_mask, pocket_ctx, batch_size, device,
                    )
                    x_prev_gen = z_x + (grad_x + grad_x_2) * 0.5 * dt_x
                    h_prev_gen = z_h + (grad_h + grad_h_2) * 0.5 * dt_h
    
                # FIX 4: Steric repulsion
                if (repulsion_scale > 0
                        and inpaint_mask.any()
                        and step_i >= int(n_steps * (1.0 - repulsion_start_frac))):
                    x_prev_gen = self._apply_repulsion(
                        x_prev_gen, inpaint_mask, lig_mask, batch_size,
                        cutoff=repulsion_cutoff,
                        scale=repulsion_scale,
                    )
    
                # B-D. Merge fixed atoms
                if inpaint_mask.any():
                    x_lig_0_current = x_lig_0_frozen - frame_offset[lig_mask]
    
                    x_prev_inp = (x_lig_0_current
                                + sigma_x_nxt * torch.randn_like(x_lig_0_current))
                    h_prev_inp = (h_lig_0_frozen
                                + sigma_h_nxt * torch.randn_like(h_lig_0_frozen))
    
                    # FIX 2: Shift GENERATED → GT frame (reversed)
                    zero_buf = torch.zeros(batch_size, 3, device=device,
                                        dtype=z_x.dtype)
                    com_gen = scatter_mean(
                        x_prev_gen[inpaint_mask],
                        lig_mask[inpaint_mask], dim=0,
                        out=zero_buf.clone(),
                    )
                    com_inp = scatter_mean(
                        x_prev_inp[inpaint_mask],
                        lig_mask[inpaint_mask], dim=0,
                        out=zero_buf.clone(),
                    )
                    delta = com_inp - com_gen
                    x_prev_gen = x_prev_gen + delta[lig_mask]
    
                    x_combined = torch.where(
                        inpaint_mask.unsqueeze(-1), x_prev_inp, x_prev_gen
                    )
                    h_combined = torch.where(
                        inpaint_mask.unsqueeze(-1), h_prev_inp, h_prev_gen
                    )
                else:
                    x_combined = x_prev_gen
                    h_combined = h_prev_gen
    
                # E. Per-step resampling re-noise
                if k < resamplings:
                    std_x = (sigma_x_cur**2 - sigma_x_nxt**2).clamp(min=0.).sqrt()
                    std_h = (sigma_h_cur**2 - sigma_h_nxt**2).clamp(min=0.).sqrt()
    
                    x_combined = x_combined + std_x * torch.randn_like(x_combined)
                    h_combined = h_combined + std_h * torch.randn_like(h_combined)

                    if inpaint_mask.any():
                        x_lig_0_current = x_lig_0_frozen - frame_offset[lig_mask]
                        x_combined[inpaint_mask] = (
                            x_lig_0_current[inpaint_mask]
                            + sigma_x_cur
                            * torch.randn_like(x_lig_0_current[inpaint_mask])
                        )
                        h_combined[inpaint_mask] = (
                            h_lig_0_frozen[inpaint_mask]
                            + sigma_h_cur
                            * torch.randn_like(h_lig_0_frozen[inpaint_mask])
                        )
                    # FIX 3: fixed-atom-only re-centering
                    com_e = self._fixed_atom_com(
                        x_combined, lig_mask, inpaint_mask, batch_size
                    )
                    x_combined = x_combined - com_e[lig_mask]
                    x_pocket   = x_pocket   - com_e[pocket_mask]
                    frame_offset = frame_offset + com_e
    
                z_x = x_combined
                z_h = h_combined
            # ── end resampling inner loop ────────────────────────────
    
            # F. Re-centre
            com_f = self._fixed_atom_com(z_x, lig_mask, inpaint_mask,
                                        batch_size)
            z_x      = z_x      - com_f[lig_mask]
            x_pocket = x_pocket - com_f[pocket_mask]
            frame_offset = frame_offset + com_f
    
            current_sigma_idx = step_i + 1
        # ── end schedule execution ───────────────────────────────────────
    
        # ── 7. Unnormalize and finalise ──────────────────────────────────
        if torch.isnan(z_x).any() or torch.isnan(z_h).any():
            raise RuntimeError(
                "[Diffusion.inpaint] NaN in final z_x/z_h."
            )
    
        x_final, h_final_cont = self.unnormalize(z_x, z_h)
    
        if inpaint_mask.any():
            com_fixed_final = scatter_mean(
                x_final[inpaint_mask], lig_mask[inpaint_mask],
                dim=0, dim_size=batch_size,
            )
            com_fixed_orig = scatter_mean(
                x_lig_original[inpaint_mask], lig_mask[inpaint_mask],
                dim=0, dim_size=batch_size,
            )
            shift = com_fixed_orig - com_fixed_final
        else:
            shift = centering_com_orig
    
        x_final = x_final + shift[lig_mask]
        x_pocket_out = x_pocket * self.norm_values[0] + shift[pocket_mask]
    
        if inpaint_mask.any():
            x_final[inpaint_mask] = x_lig_original[inpaint_mask]
    
        h_final = F.one_hot(
            torch.argmax(h_final_cont, dim=-1), num_classes=self.atom_nf
        ).float()
        if inpaint_mask.any():
            h_final[inpaint_mask] = h_lig_original[inpaint_mask]
    
        xh_pocket_out = torch.cat([x_pocket_out, h_pocket], dim=-1)
        xh_lig_out = torch.cat([x_final, h_final], dim=-1)
    
        return xh_lig_out, xh_pocket_out, lig_mask, pocket_mask
    
    
    # ──────────────────────────────────────────────────────────────────────────
    # Helper: Fixed-atom CoM (same as v1)
    # ──────────────────────────────────────────────────────────────────────────
    
    def _fixed_atom_com(self, x, lig_mask, inpaint_mask, batch_size):
        """Per-batch CoM over fixed atoms; falls back to all-atom CoM."""
        if inpaint_mask.any():
            x_fix = torch.zeros_like(x)
            x_fix[inpaint_mask] = x[inpaint_mask]
            n_fix = scatter_add(
                inpaint_mask.float(), lig_mask, dim=0,
                dim_size=batch_size,
            )
            sum_fix = scatter_add(x_fix, lig_mask, dim=0,
                                dim_size=batch_size)
            com = sum_fix / n_fix.unsqueeze(-1).clamp(min=1)
    
            has_fix = n_fix > 0
            if not has_fix.all():
                com_all = scatter_mean(x, lig_mask, dim=0,
                                    dim_size=batch_size)
                com[~has_fix] = com_all[~has_fix]
            return com
        else:
            return scatter_mean(x, lig_mask, dim=0, dim_size=batch_size)
    
    
    # ──────────────────────────────────────────────────────────────────────────
    # Helper: Steric repulsion (same as v1)
    # ──────────────────────────────────────────────────────────────────────────
    
    def _apply_repulsion(self, x_gen, inpaint_mask, lig_mask, batch_size,
                        cutoff=1.2, scale=0.01):
        """Soft repulsive push on free atoms too close to fixed atoms."""
        free_mask = ~inpaint_mask
        if not free_mask.any() or not inpaint_mask.any():
            return x_gen
    
        x_free  = x_gen[free_mask]
        x_fixed = x_gen[inpaint_mask]
        batch_free  = lig_mask[free_mask]
        batch_fixed = lig_mask[inpaint_mask]
    
        correction = torch.zeros_like(x_free)
    
        for b in range(batch_size):
            sel_f = (batch_free == b)
            sel_k = (batch_fixed == b)
            if not sel_f.any() or not sel_k.any():
                continue
    
            xf = x_free[sel_f]
            xk = x_fixed[sel_k]
    
            diff = xf.unsqueeze(1) - xk.unsqueeze(0)
            dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    
            within = (dist.squeeze(-1) < cutoff)
            if not within.any():
                continue
    
            mag = scale * (cutoff - dist) / (dist ** 2 + 1e-6)
            direction = diff / dist
            push = (mag * direction * within.unsqueeze(-1).float()).sum(dim=1)
            correction[sel_f] = push
    
        x_out = x_gen.clone()
        x_out[free_mask] = x_out[free_mask] + correction
        return x_out

class DistributionNodes:
    def __init__(self, histogram):

        histogram = torch.tensor(histogram).float()
        histogram = histogram + 1e-3  # for numerical stability

        prob = histogram / histogram.sum()

        self.idx_to_n_nodes = torch.tensor(
            [[(i, j) for j in range(prob.shape[1])] for i in range(prob.shape[0])]
        ).view(-1, 2)

        self.n_nodes_to_idx = {tuple(x.tolist()): i
                               for i, x in enumerate(self.idx_to_n_nodes)}

        self.prob = prob
        self.m = torch.distributions.Categorical(self.prob.view(-1),
                                                 validate_args=True)

        self.n1_given_n2 = \
            [torch.distributions.Categorical(prob[:, j], validate_args=True)
             for j in range(prob.shape[1])]
        self.n2_given_n1 = \
            [torch.distributions.Categorical(prob[i, :], validate_args=True)
             for i in range(prob.shape[0])]

        # entropy = -torch.sum(self.prob.view(-1) * torch.log(self.prob.view(-1) + 1e-30))
        entropy = self.m.entropy()
        print("Entropy of n_nodes: H[N]", entropy.item())

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        num_nodes_lig, num_nodes_pocket = self.idx_to_n_nodes[idx].T
        return num_nodes_lig, num_nodes_pocket

    def sample_conditional(self, n1=None, n2=None):
        assert (n1 is None) ^ (n2 is None), \
            "Exactly one input argument must be None"

        m = self.n1_given_n2 if n2 is not None else self.n2_given_n1
        c = n2 if n2 is not None else n1

        return torch.tensor([m[i].sample() for i in c], device=c.device)

    def log_prob(self, batch_n_nodes_1, batch_n_nodes_2):
        assert len(batch_n_nodes_1.size()) == 1
        assert len(batch_n_nodes_2.size()) == 1

        idx = torch.tensor(
            [self.n_nodes_to_idx[(n1, n2)]
             for n1, n2 in zip(batch_n_nodes_1.tolist(), batch_n_nodes_2.tolist())]
        )

        # log_probs = torch.log(self.prob.view(-1)[idx] + 1e-30)
        log_probs = self.m.log_prob(idx)

        return log_probs.to(batch_n_nodes_1.device)

    def log_prob_n1_given_n2(self, n1, n2):
        assert len(n1.size()) == 1
        assert len(n2.size()) == 1
        log_probs = torch.stack([self.n1_given_n2[c].log_prob(i.cpu())
                                 for i, c in zip(n1, n2)])
        return log_probs.to(n1.device)

    def log_prob_n2_given_n1(self, n2, n1):
        assert len(n2.size()) == 1
        assert len(n1.size()) == 1
        log_probs = torch.stack([self.n2_given_n1[c].log_prob(i.cpu())
                                 for i, c in zip(n2, n1)])
        return log_probs.to(n2.device)

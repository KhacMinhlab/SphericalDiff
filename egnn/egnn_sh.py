"""
@inproceedings{Batatia2022mace,
  title={{MACE}: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},
  author={Ilyes Batatia and David Peter Kovacs and Gregor N. C. Simm and Christoph Ortner and Gabor Csanyi},
  booktitle={Advances in Neural Information Processing Systems},
  editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
  year={2022},
  url={https://openreview.net/forum?id=YPpSngE-ZU}
}

@misc{Batatia2022Design,
  title = {The Design Space of E(3)-Equivariant Atom-Centered Interatomic Potentials},
  author = {Batatia, Ilyes and Batzner, Simon and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Musaelian, Albert and Simm, Gregor N. C. and Drautz, Ralf and Ortner, Christoph and Kozinsky, Boris and Cs{\'a}nyi, G{\'a}bor},
  year = {2022},
  number = {arXiv:2205.06643},
  eprint = {2205.06643},
  eprinttype = {arxiv},
  doi = {10.48550/arXiv.2205.06643},
  archiveprefix = {arXiv}
 }
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.o3 import TensorProduct
from torch_scatter import scatter_mean
from typing import Any, Optional, Tuple
import math
from mace.modules import EquivariantProductBasisBlock
from mace.modules.blocks import InteractionBlock
from mace.modules.irreps_tools import tp_out_irreps_with_instructions, reshape_irreps
from mace.modules.wrapper_ops import (
    CuEquivarianceConfig,
    FullyConnectedTensorProduct,
    Linear,
    TensorProduct
)
from mace.tools.scatter import scatter_sum
from egnn.radial import BesselBasis, GaussianSmearing, PolynomialCutoff, ExpNormalSmearing, RadialMLP

class EquivariantFiLM(nn.Module):
    """FiLM conditioning that respects equivariance"""
    def __init__(self, hidden_nf, irreps):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        
        # Count number of channels per L
        self.num_channels = sum(mul for mul, ir in self.irreps)
        
        self.time_to_film = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, self.num_channels * 2)  # scale + shift per CHANNEL
        )
        nn.init.constant_(self.time_to_film[-1].weight, 0)
        nn.init.constant_(self.time_to_film[-1].bias, 0)
        
        # Build index map for applying scale/shift per channel
        self._build_index_map()
    
    def _build_index_map(self):
        """Maps each element to its channel index"""
        self.channel_to_elements = []
        idx = 0
        for mul, ir in self.irreps:
            for m in range(mul):
                self.channel_to_elements.append((idx, idx + ir.dim))
                idx += ir.dim
        
        # Precompute for vectorized ops
        self.element_to_channel = []
        for c_idx, (start, end) in enumerate(self.channel_to_elements):
            self.element_to_channel.extend([c_idx] * (end - start))
        self.register_buffer('_channel_idx', 
                           torch.tensor(self.element_to_channel, dtype=torch.long))
        
        # Mask for shift (only L=0)
        shift_mask = []
        for mul, ir in self.irreps:
            for m in range(mul):
                if ir.l == 0:
                    shift_mask.extend([1.0] * ir.dim)
                else:
                    shift_mask.extend([0.0] * ir.dim)
        self.register_buffer('shift_mask', torch.tensor(shift_mask))
    
    def forward(self, features, t_emb):
        """
        features: [N, irreps_dim]
        t_emb: [N, hidden_nf]
        """
        film_params = self.time_to_film(t_emb)  # [N, num_channels * 2]
        scale_per_channel, shift_per_channel = film_params.chunk(2, dim=-1)
        
        # Expand to per-element
        scale = scale_per_channel[:, self._channel_idx]  # [N, irreps_dim]
        shift = shift_per_channel[:, self._channel_idx]  # [N, irreps_dim]
        
        # Apply with L>0 shift masking
        return features * (1.0 + scale) + shift * self.shift_mask

class SeparableLayerNorm(nn.Module):
    def __init__(self, irreps_in: o3.Irreps, eps: float = 1e-3): 
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.eps = eps
     
        self.dim_list = []
        self.mul_list = []
        
        # Tính tổng số kênh vector
        count_scalar = 0
        count_vec_elements = 0
        
        for mul, ir in self.irreps_in:
            if ir.l == 0:
                count_scalar += mul
            else:
                self.dim_list.append(ir.dim) # [3, 5, 7...]
                self.mul_list.append(mul)
                count_vec_elements += mul * ir.dim
        
        # --- Params cho Scalar ---
        if count_scalar > 0:
            self.gamma_scalar = nn.Parameter(torch.ones(count_scalar))
            self.beta_scalar = nn.Parameter(torch.zeros(count_scalar))
        else:
            self.register_parameter('gamma_scalar', None)
            self.register_parameter('beta_scalar', None)

        self.vector_gammas = nn.ParameterList()
        num_vec_blocks = len(self.dim_list)
        if num_vec_blocks > 0:
            for mul in self.mul_list:
                self.vector_gammas.append(nn.Parameter(torch.ones(mul)))
            balance_weights = []
            for dim, mul in zip(self.dim_list, self.mul_list):
                w = 1.0 / dim
                balance_weights.extend([w] * (dim * mul))
            
            total_w = sum(balance_weights)
            self.register_buffer('balance_weights', torch.tensor(balance_weights))
        
        self.scalar_idx = []
        self.vector_idx = []
        curr = 0
        for mul, ir in self.irreps_in:
            length = mul * ir.dim
            if ir.l == 0:
                self.scalar_idx.append((curr, curr + length))
            else:
                self.vector_idx.append((curr, curr + length, mul, ir.dim))
            curr += length

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_out = x.clone()
        
        if self.gamma_scalar is not None:
            scalars = []
            for start, end in self.scalar_idx:
                scalars.append(x[:, start:end])
            
            if scalars:
                s_cat = torch.cat(scalars, dim=-1)
                mean = s_cat.mean(dim=-1, keepdim=True)
                var = s_cat.var(dim=-1, keepdim=True, unbiased=False)
                s_norm = (s_cat - mean) / torch.sqrt(var + self.eps)
                s_out = s_norm * self.gamma_scalar + self.beta_scalar
                
                offset = 0
                for start, end in self.scalar_idx:
                    size = end - start
                    x_out[:, start:end] = s_out[:, offset:offset+size]
                    offset += size

        # --- 2. Xử lý Vector (Weighted Joint RMSNorm) ---
        if len(self.vector_idx) > 0:
            # Gom toàn bộ vector lại
            vec_parts = []
            for start, end, _, _ in self.vector_idx:
                vec_parts.append(x[:, start:end])
            
            if vec_parts:
                all_vecs = torch.cat(vec_parts, dim=-1) # [Batch, Total_Vec_Dim]
                
                sq_vecs = torch.clamp(all_vecs, min=-1e4, max=1e4).pow(2)
                weighted_sq = sq_vecs * self.balance_weights.unsqueeze(0)
                
                mean_sq = weighted_sq.sum(dim=-1, keepdim=True) / self.balance_weights.sum()

                rms = torch.sqrt(torch.clamp(mean_sq, min=1e-6) + self.eps)
                
                current_offset = 0
                for i, (start, end, mul, dim) in enumerate(self.vector_idx):
                    vec_block = x[:, start:end]
                    
                    vec_norm = vec_block / rms
                    
                    gamma = self.vector_gammas[i]
                    vec_norm = vec_norm.view(x.shape[0], mul, dim)
                    vec_norm = vec_norm * gamma.view(1, mul, 1)
                    
                    x_out[:, start:end] = vec_norm.reshape(x.shape[0], -1)

        return x_out

class MACEInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"): self.cueq_config = None
        if not hasattr(self, "oeq_config"): self.oeq_config = None 

        self.linear_up = Linear(
                self.node_feats_irreps,
                self.edge_irreps,
                internal_weights=True,
                shared_weights=True,
                cueq_config=self.cueq_config,
            )
        
        node_scalar_irreps = o3.Irreps(
            [(self.node_feats_irreps.count(o3.Irrep(0, 1)), (0, 1))]
        )

        self.source_embedding = Linear(
            self.node_attrs_irreps,
            node_scalar_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        self.target_embedding = Linear(
            self.node_attrs_irreps,
            node_scalar_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        torch.nn.init.uniform_(self.source_embedding.weight, a=-0.1, b=0.1)
        torch.nn.init.uniform_(self.target_embedding.weight, a=-0.1, b=0.1)

        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.edge_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )

        self.conv_tp = TensorProduct(
            self.edge_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
            oeq_config=self.oeq_config,
        )

        input_dim = self.edge_feats_irreps.num_irreps
        
        self.conv_tp_weights = RadialMLP([input_dim + 2 * node_scalar_irreps.dim] + 
                                         self.radial_MLP + [self.conv_tp.weight_numel])
        self.irreps_out = self.target_irreps

        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        self.skip_tp = FullyConnectedTensorProduct(
            self.node_feats_irreps,
            self.node_attrs_irreps,
            self.hidden_irreps,
            cueq_config=self.cueq_config,
        )
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Optional[torch.Tensor] = None,
        lammps_class: Optional[Any] = None,
        lammps_natoms: Tuple[int, int] = (0, 0),
        first_layer: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        sc = self.skip_tp(node_feats, node_attrs) # Residual connection
        node_feats = self.linear_up(node_feats)
        
        source_emb = self.source_embedding(node_attrs) # [N_nodes, embed_dim]
        target_emb = self.target_embedding(node_attrs) # [N_nodes, embed_dim]

        edge_feats_augmented = torch.cat([
            edge_feats,
            source_emb[edge_index[0]],
            target_emb[edge_index[1]]
        ], dim=-1)
        
        # Sinh weights từ RadialMLP custom
        tp_weights = self.conv_tp_weights(edge_feats_augmented)
        
        if cutoff is not None:
            tp_weights = tp_weights * cutoff

        # Convolution
        mji = self.conv_tp(
            node_feats[edge_index[0]], edge_attrs, tp_weights
        )
        message = scatter_sum(
            src=mji, index=edge_index[1], dim=0, dim_size=node_feats.shape[0]
        )
        
        # Linear update
        message = self.linear(message) / self.avg_num_neighbors
        
        return self.reshape(message), sc
    
class MACEBlock(nn.Module):

    def __init__(
        self,
        irreps_node_input: o3.Irreps,
        irreps_node_hidden: o3.Irreps, 
        irreps_sh: o3.Irreps,
        irreps_node_attrs: o3.Irreps,
        max_correlation_order=3,  # 4-body interactions by default
        edge_feat_input_dim: int = 16, 
        avg_num_neighbors: float = 10.0,
        num_elements: int = None
    ):
        super().__init__()
        self.max_correlation_order = max_correlation_order
                
        self.interaction = MACEInteractionBlock(
            node_feats_irreps=irreps_node_input,
            target_irreps=irreps_node_hidden,
            hidden_irreps=irreps_node_hidden,
            avg_num_neighbors=avg_num_neighbors,
            edge_attrs_irreps=irreps_sh,
            edge_feats_irreps=o3.Irreps(f"{edge_feat_input_dim}x0e"), 
            node_attrs_irreps=irreps_node_attrs,
            radial_MLP=[256, 512, 1024]
        )     
        # Bước 2: Khối Many-Body (Product Basis) để tạo message cuối cùng
        self.product_basis = EquivariantProductBasisBlock(
            node_feats_irreps=irreps_node_hidden, # Đầu vào là A-features
            target_irreps=irreps_node_hidden,      # Đầu ra là message
            correlation=max_correlation_order,
            use_sc=True,
            use_agnostic_product=False,
            num_elements=num_elements
        )

        self._is_radial_initialized = False
    
    
    def forward(self, h,node_attrs, edge_index, edge_sh, edge_feats, batch_mask=None, cutoff=None,**kwargs):
    
        a_features, sc = self.interaction(
            node_feats=h,
            node_attrs=node_attrs,
            edge_attrs=edge_sh,
            edge_feats=edge_feats,
            edge_index=edge_index,
            cutoff=cutoff
        )

        # Bước 2: Tạo message cuối cùng từ tương tác many-body
        messages = self.product_basis(
            node_feats=a_features,
            sc=sc, # Truyền skip connection từ bước tương tác
            node_attrs=node_attrs
        )
        max_msg = messages.abs().max().item()


        return messages
    
class ScalarOutputHead(nn.Module):
    def __init__(self, irreps_in: o3.Irreps, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        
        # 1. Chỉ lọc lấy phần Scalar (L=0)
        # MACE gốc chỉ dùng features L=0 cho Readout 
        self.irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in self.irreps_in if ir.l == 0])
        
        # Kiểm tra xem có scalar không
        if self.irreps_scalars.dim == 0:
            raise ValueError("Input features must contain L=0 scalars for readout.")

        # 2. Linear projection: Tự động trích xuất L=0 và bỏ qua L>0
        # e3nn Linear rất thông minh, nó chỉ kết nối các irreps tương thích.
        # Khi chiếu từ (L=0 + L=1 + ...) sang (L=0), nó tự động cắt bỏ L>0.
        self.extract_scalars = o3.Linear(self.irreps_in, self.irreps_scalars)
        
        # 3. MLP đầu ra
        self.output_mlp = nn.Sequential(
            nn.Linear(self.irreps_scalars.dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Bước 1: Trích xuất scalars (L=0) - e3nn làm việc này cực nhanh
        scalars = self.extract_scalars(features)
        
        # Bước 2: Dự đoán
        return self.output_mlp(scalars)

class VectorOutputHead(nn.Module):
    def __init__(self, irreps_in: o3.Irreps, hidden_dim: int = 64):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        
        # Tách rời definition để control init
        self.irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in self.irreps_in if ir.l == 0])
        self.irreps_vectors = o3.Irreps("1x1o")

        # 1. Project Vectors (Cần init nhỏ)
        self.vector_proj = o3.Linear(self.irreps_in, self.irreps_vectors, biases=False)
        
        # 2. Extract Scalars (Cần init bình thường để MLP học được)
        self.scalar_extract = o3.Linear(self.irreps_in, self.irreps_scalars)
        
        # 3. MLP dự đoán Scale
        self.scale_mlp = nn.Sequential(
            nn.Linear(self.irreps_scalars.dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        # CHỈ thu nhỏ phần vector projection
        with torch.no_grad():
            self.vector_proj.weight.data.mul_(0.01)
            
        # Phần scalar extract GIỮ NGUYÊN (hoặc init xavier mặc định)
        
        # Init lớp cuối của MLP về 0 để scale bắt đầu từ 1.0 (nhưng gradient vẫn chảy được qua các lớp trước)
        nn.init.constant_(self.scale_mlp[-1].weight, 0)
        nn.init.constant_(self.scale_mlp[-1].bias, 0.0) 

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Gọi riêng 2 path
        vec_slice = self.vector_proj(features)
        scalar_slice = self.scalar_extract(features)
        
        raw_scale = self.scale_mlp(scalar_slice)
        scale = 1.0 + raw_scale 
        
        return vec_slice * scale

class EGNN_Spherical(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf    , device ='cpu',
                 act_fn=nn.SiLU(), n_layers = 3, out_node_nf=None,
                 lmax=2, num_rbf=16,
                 max_radius=10.0, 
                 rbf="expnormal", trainable_rbf=True, 
                 max_correlation_order =3,
                 weight_mode='dynamic',
                 use_reg_loss=False,
                 num_atom_types=None, avg_num_neighbors=10.0):
        super().__init__()
        if num_atom_types is None:
            num_atom_types = in_node_nf
            
        if out_node_nf is None:
            out_node_nf = in_node_nf

        self.num_elements = num_atom_types


        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.lmax = lmax
        self.max_radius = max_radius
        self.num_rbf = num_rbf
        self.weight_mode = weight_mode

        self.use_reg_loss = use_reg_loss
        self.cutoff_fn = PolynomialCutoff(r_max=max_radius, p=6)
        
        muls = []
        for l in range(lmax + 1):
            mul = hidden_nf 
            muls.append(f"{mul}x{l}{'e' if l % 2 == 0 else 'o'}")
        
        irreps_string = "+".join(muls)

        irreps_hidden = o3.Irreps(irreps_string)
        self.irreps_hidden_dim = irreps_hidden.dim
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)

        shift_mask = []
        for mul, ir in irreps_hidden:
            if ir.l == 0:
                shift_mask.extend([1.0] * mul * ir.dim)
            else:
                shift_mask.extend([0.0] * mul * ir.dim)
        
        self.register_buffer('shift_mask', torch.tensor(shift_mask))
        # Edge embedding
        self.edge_embed_dim = num_rbf + in_edge_nf + hidden_nf

        if rbf == 'bessel':
            self.radial_basis = BesselBasis(
                r_max=max_radius, num_basis=num_rbf, trainable=trainable_rbf
                )

        elif rbf == 'expnormal':
            self.radial_basis = ExpNormalSmearing(
                cutoff=max_radius, num_rbf=num_rbf, trainable=trainable_rbf
            )
        else:
            self.radial_basis = GaussianSmearing(0.0, max_radius, num_rbf)

        #Input Embedding

        self.embedding = nn.Sequential(
            nn.Linear(in_node_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf)
        )

        # Project to full irreps
        self.scalar_to_irreps = Linear(
            o3.Irreps(f"{hidden_nf}x0e"),
            irreps_hidden
        )

        self.attr_embed_dim = hidden_nf // 2

        self.irreps_node_attrs = o3.Irreps(f"{self.num_elements}x0e")

        self.layers = nn.ModuleList()
        self.coord_heads = nn.ModuleList()
        self.norms = nn.ModuleList()

        current_irreps = irreps_hidden

        self.film_layer = EquivariantFiLM(hidden_nf, irreps_hidden)

        for _ in range(n_layers):
                block = MACEBlock(
                    irreps_node_input=current_irreps,
                    irreps_node_hidden=irreps_hidden,
                    irreps_sh=self.irreps_sh,
                    irreps_node_attrs=self.irreps_node_attrs,
                    max_correlation_order=max_correlation_order,
                    edge_feat_input_dim=self.edge_embed_dim,
                    avg_num_neighbors=avg_num_neighbors,
                    num_elements=self.num_elements
                )                              
                
                self.layers.append(block)
                self.norms.append(SeparableLayerNorm(current_irreps))
                self.coord_heads.append(VectorOutputHead(current_irreps))

        # Output heads
        self.final_norm = SeparableLayerNorm(current_irreps)
        self.output_head = ScalarOutputHead(
            current_irreps, out_node_nf, hidden_dim=hidden_nf
        )

    def compute_edge_features(self, x, edge_index, rbf_scale=None):

        src, dst = edge_index

        rel_pos = x[src] - x[dst]
        edge_length = torch.sqrt(
            torch.sum(rel_pos ** 2, dim=1, keepdim=True) + 1e-8
        ) 

        edge_sh = o3.spherical_harmonics(
            self.irreps_sh,
            rel_pos,
            normalize=True,
            normalization='component'
        )

        edge_length_embedded = self.radial_basis(edge_length.squeeze(-1))
        cutoff = self.cutoff_fn(edge_length)

        return edge_sh, edge_length_embedded, cutoff  
    
    def remove_com_update(self, coord_update, batch_mask):

        mean_update = scatter_mean(coord_update, batch_mask, dim=0)
        
        return coord_update - mean_update[batch_mask]

    
    def forward(self, h, x, edge_index, t, node_attrs, node_mask=None, edge_mask=None, 
                update_coords_mask=None, batch_mask=None, edge_attr=None, time_emb=None, rbf_scale=None, debugger=None, scale_tracker=None):
        """
        Forward pass
        
        Args:
            h: [N, in_node_nf] - node features
            x: [N, 3] - node coordinates
            edge_index: [2, E] - edge indices
            node_mask: [N, 1] - node mask
            edge_mask: [E, 1] - edge mask
            update_coords_mask: [N, 1] - mask for coordinate updates
            batch_mask: [N] - batch indices
            edge_attr: [E, in_edge_nf] - edge attributes
            
        Returns:
            h_out: [N, out_node_nf] - output node features
            x_out: [N, 3] - output coordinates
        """
        if time_emb is not None:
            t_emb = time_emb

        if batch_mask is not None:
            t_emb_atom = t_emb[batch_mask]
        else:
            t_emb_atom = t_emb.repeat(h.shape[0], 1)


        h_embedded = self.embedding(h)
        h_irreps= self.scalar_to_irreps(h_embedded)

        x_current = x.clone()
        total_coord_update = torch.zeros_like(x)

        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        t_emb_edge = t_emb_atom[src_idx]

                 
        for i, layer in enumerate(self.layers):

            edge_sh, edge_length_embedded, cutoff = self.compute_edge_features(x_current, edge_index)
            # Embed input
            if edge_attr is not None:
                # Tổng kích thước = num_rbf + edge_embedding_dim
                edge_features = torch.cat([edge_length_embedded, edge_attr, t_emb_edge], dim=-1)
            else:
                edge_features = torch.cat([edge_length_embedded, t_emb_edge], dim=-1)


            h_normed = self.norms[i](h_irreps)

            h_working = self.film_layer(h_normed, t_emb_atom)

            new_h_irreps = layer(
                h=h_working,
                node_attrs=node_attrs,
                edge_index=edge_index,
                edge_sh=edge_sh,
                edge_feats=edge_features, 
                time_emb=t_emb,
                batch_mask=batch_mask,
                cutoff=cutoff
            )

            h_for_coord_update = h_working + new_h_irreps
            delta_x_spherical = self.coord_heads[i](h_for_coord_update)

            delta_x_layer = delta_x_spherical
            total_coord_update = total_coord_update + delta_x_layer

            if update_coords_mask is not None:
                x_current = x_current + delta_x_layer * update_coords_mask
            else:
                x_current = x_current + delta_x_layer

            h_irreps = h_irreps + new_h_irreps

        h_final = self.final_norm(h_irreps)

        h_out = self.output_head(h_final)

        if update_coords_mask is not None:
            total_coord_update = total_coord_update * update_coords_mask

        if batch_mask is not None:
            total_coord_update = self.remove_com_update(total_coord_update, batch_mask)

        x_out = x + total_coord_update

        if node_mask is not None:
            h_out = h_out * node_mask
            x_out = x_out * node_mask
        return h_out, x_out


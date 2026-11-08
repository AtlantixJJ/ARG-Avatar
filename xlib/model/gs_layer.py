import torch
import torch.nn as nn
from typing import Union

from xlib.render.gaussian_splatting import GaussianParameters, RGB2SH, SH2RGB, get_gaussian_type
from xlib.basetype import TensorDict
from xlib.geometry.ops import inverse_sigmoid, trunc_exp, quaternion_multiply


"""
gs_tpose_dic: canonical GS representation
1. xyz: anchor points, from neutral mesh or UV interpolated points
2. offset_xyz: offset added to xyz (neutral m)
3. scale: activated scale
4. rotation: activated rotation
5. opacity: activated opacity
6. sh: activated SH coefficients
7. scale_: unactivated scale
8. opacity_: unactivated opacity
9. sh_: unactivated SH coefficients
10. rotation_: we shouldn't have this, rotation is always activated (normalized)

gs_posed:
1. xyz: posed vertices
2. scale: posed scale
3. rotation: posed rotation
4. opacity: posed opacity
5. sh: posed SH coefficients
no unactivated values
"""

def tpose_to_raw_keys(gs_tpose_dic: dict) -> dict:
    ret = {
        "offset_xyz": gs_tpose_dic["offset_xyz"],
        "scale_": gs_tpose_dic["scale_"],
        "rotation": gs_tpose_dic["rotation"],
        "opacity_": gs_tpose_dic["opacity_"],
        "sh_": gs_tpose_dic["sh_"],
    }
    if "offset_pose" in gs_tpose_dic:
        ret["offset_pose"] = gs_tpose_dic["offset_pose"]
    return ret


def raw_to_tpose_keys(raw_gs_dic: dict) -> dict:
    offset_xyz = raw_gs_dic["offset_xyz"] if "offset_xyz" in raw_gs_dic else raw_gs_dic["xyz"]
    ret = {
        "offset_xyz": offset_xyz,
        "scale_": raw_gs_dic["scale_"],
        "rotation": raw_gs_dic["rotation"],
        "opacity_": raw_gs_dic["opacity_"],
        "sh_": raw_gs_dic["sh_"],
    }
    if "offset_pose" in raw_gs_dic:
        ret["offset_pose"] = raw_gs_dic["offset_pose"]
    return ret
    

class GSLayer(nn.Module):
    """W/O Activation Function"""

    # Maps (activated_key, raw_key): output key -> input key for activate()
    RAW_TO_ACTIVATED = {
        'rotation': 'rotation',
        'scale_': 'scale',
        'opacity_': 'opacity',
        'sh_': 'sh',
        'offset_xyz': 'offset_xyz',
    }

    def setup_functions(self):

        self.xyz_offset_activation = lambda x: torch.tanh(x) * self.xyz_offset_max_step
        self.inverse_xyz_offset_activation = lambda x: torch.atanh(torch.clamp(x / self.xyz_offset_max_step, -0.999999, 0.999999))
   

        # distance-based scaling is active when scale_base is provided.
        self.scaling_activation = self._scaling_activation
        self.scaling_inverse_activation = self._scaling_inverse_activation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = lambda x: torch.nn.functional.normalize(x, dim=-1, eps=1e-8)

        self.rgb_activation = torch.sigmoid
        self.inverse_rgb_activation = inverse_sigmoid

    def __init__(
        self,
        in_channels,
        n_layers=1,
        hidden_channels=128,
        clip_scaling=0.2,
        init_scaling=0.0,
        init_density=0.99,
        sh_degree=0,
        xyz_offset=True,
        xyz_offset_max_step=1.0,
        fix_opacity=False,
        fix_rotation=False,
        isotropic_gaussians=False,
        scale_base_factor=1.0,
    ):
        super().__init__()

        self.scale_base_factor = scale_base_factor
        self.clip_scaling = clip_scaling
        self.n_layers = n_layers

        self.xyz_offset = xyz_offset
        self.sh_degree = sh_degree
        self.xyz_offset_max_step = xyz_offset_max_step  # 1.2 / 32
        self.fix_opacity = fix_opacity
        self.fix_rotation = fix_rotation
        self.isotropic_gaussians = isotropic_gaussians
        self.register_buffer("scale_base", None, persistent=False)

        gaussian_type = get_gaussian_type()
        scale_dim = 3 if gaussian_type == "3d" else 2

        self.attr_dict = {
            "sh": (sh_degree + 1) ** 2 * 3,
            "scale": scale_dim,
            "offset_xyz": 3,
            "opacity": None,
            "rotation": 4,
        }
        if not self.fix_opacity:
            self.attr_dict["opacity"] = 1

        # Keep the direct-decode path aligned with the learned-head path.
        # When n_layers == 0 the upstream decoder emits raw residual channels,
        # so we still need the same per-attribute initialization offsets that
        # the explicit Linear heads would have contributed via their biases.
        self.raw_attr_bias = {
            "offset_xyz": None,
            "scale_": init_scaling,
            "rotation": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "opacity_": 0.0 if self.fix_opacity else inverse_sigmoid(init_density),
            "sh_": None,
        }

        self.out_layers = nn.ModuleDict()
        if n_layers == 0:
            self.setup_functions()
            return

        for key, out_ch in self.attr_dict.items():
            if out_ch is None:
                continue

            layers = []
            if n_layers > 1:
                layers.append(nn.Linear(in_channels, hidden_channels))
                layers.append(nn.GELU())
                for i in range(n_layers - 2):
                    layers.append(nn.Linear(hidden_channels, hidden_channels))
                    layers.append(nn.GELU())
                h_channels = hidden_channels
            else:
                h_channels = in_channels
            if key == "sh":
                out_ch = (self.sh_degree + 1) ** 2 * 3
            layers.append(nn.Linear(h_channels, out_ch))
            layer = layers[-1]

            # initialize
            if key == "sh" and self.sh_degree > 0:
                with torch.no_grad():
                    layer.bias[3:].zero_()
            
            if key != "sh":
                if key == "opacity" and self.fix_opacity:
                    pass
                elif key == "rotation" and self.fix_rotation:
                    pass
                else:
                    nn.init.constant_(layer.weight, 0)
                    nn.init.constant_(layer.bias, 0)
            if key == "scale":
                nn.init.constant_(layer.bias, init_scaling)
            elif key == "rotation":
                if not self.fix_rotation:
                    nn.init.constant_(layer.bias, 0)
                    nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                if not self.fix_opacity:
                    nn.init.constant_(layer.bias, inverse_sigmoid(init_density))
            self.out_layers[key] = nn.Sequential(*layers)

        self.setup_functions()

    def hyper_step(self, step):
        self.clip_scaling = self.clip_scaling_pruner.get_value(step)

    def _scaling_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale_base is not None:
            return torch.sigmoid(x) * self.scale_base
        return trunc_exp(x)

    def _scaling_inverse_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale_base is not None:
            return inverse_sigmoid(x / self.scale_base)
        return torch.log(x)

    def constrain_forward(self, ret, constrain_dict):

        # body scaling constrain
        # gs_attr.scaling[is_constrain_body] = gs_attr.scaling[is_constrain_body].clamp(max=0.02)  # magic number, which is used to constrain 
        # hand opacity constrain

        # force the hand's opacity to be 0.95
        # gs_attr.opacity[is_hand] = gs_attr.opacity[is_hand].clamp(min=0.95)

        # body scaling constrain
        # is_constrain_body = constrain_dict['is_constrain_body']
        is_upper_body = constrain_dict['is_upper_body']
        scaling = ret['scale'] 
        # scaling[is_constrain_body] body_constrain= scaling[is_constrain_body].clamp(max = 0.02)
        scaling[is_upper_body] = scaling[is_upper_body].clamp(max = 0.02)
        # scaling = scaling.clamp(max=0.02)
        ret['scale'] = scaling

        return ret

    def presum(self, dic: TensorDict) -> TensorDict:
        with torch.amp.autocast(device_type='cuda', enabled=False):
            return TensorDict.from_dict({
                'sh_': dic['sh_'].unsqueeze(-2),
                'rotation': self.rotation_activation(dic['rotation']),
                'offset_xyz': self.xyz_offset_activation(dic["offset_xyz"]),
                **(
                    {'offset_pose': self.xyz_offset_activation(dic["offset_pose"])}
                    if "offset_pose" in dic else {}
                ),
                'scale_': dic['scale_'],
                'opacity_': dic['opacity_'],
            })

    def forward_raw(self, x: torch.Tensor) -> TensorDict:
        #with torch.amp.autocast(device_type='cuda', enabled=False):
        x = x.float()  # Convert to fp32 since autocast is disabled
        if self.n_layers == 0:
            split_keys = []
            split_dims = []
            for k_in, k_out in self.RAW_TO_ACTIVATED.items():
                if k_out not in self.attr_dict:
                    continue
                out_ch = self.attr_dict[k_out]
                if out_ch is None:
                    out_ch = 1
                elif k_out == "sh":
                    # Keep parity with learned heads, which predict RGB logits
                    # then convert to SH in activate().
                    out_ch = (self.sh_degree + 1) ** 2 * 3
                split_keys.append(k_in)
                split_dims.append(out_ch)

            total_ch = sum(split_dims)
            if x.shape[-1] != total_ch:
                raise ValueError(
                    f"GSLayer n_layers=0 expects input dim {total_ch}, got {x.shape[-1]}"
                )
            chunks = torch.split(x, split_dims, dim=-1)
            dic = {}
            for key, chunk in zip(split_keys, chunks):
                bias = self.raw_attr_bias.get(key)
                if key == "rotation" and self.fix_rotation:
                    dic[key] = bias.to(device=chunk.device, dtype=chunk.dtype).view(
                        *([1] * (chunk.ndim - 1)), -1
                    ).expand_as(chunk)
                    continue
                if key == "opacity_" and self.fix_opacity:
                    if not torch.is_tensor(bias):
                        bias = torch.tensor(bias, device=chunk.device, dtype=chunk.dtype)
                    else:
                        bias = bias.to(device=chunk.device, dtype=chunk.dtype)
                    dic[key] = bias.view(*([1] * (chunk.ndim - 1)), -1).expand_as(chunk)
                    continue
                if bias is None:
                    dic[key] = chunk
                    continue
                if not torch.is_tensor(bias):
                    bias = torch.tensor(bias, device=chunk.device, dtype=chunk.dtype)
                else:
                    bias = bias.to(device=chunk.device, dtype=chunk.dtype)
                dic[key] = chunk + bias.view(*([1] * (chunk.ndim - 1)), -1)
            return TensorDict.from_dict(dic)

        dic = {}
        for k_in, k_out in self.RAW_TO_ACTIVATED.items():
            if k_out not in self.attr_dict:
                continue
            out_ch = self.attr_dict[k_out] 
            if k_out in self.out_layers:
                # we use activated name as parameter name
                # but the output is in raw domain, so we use k_in
                dic[k_in] = self.out_layers[k_out](x)
            else:
                # This happens for opacity and rotation when they are fixed
                if out_ch is None:  # opacity case
                    out_ch = 1
                shape = (*x.shape[:-1], out_ch)
                dic[k_in] = torch.zeros(shape, device=x.device, dtype=x.dtype)
        return TensorDict.from_dict(dic)

    def unactivate(self, dic: Union[dict, TensorDict]) -> TensorDict:
        """Only unactivate scale, opacity and sh."""
        with torch.amp.autocast(device_type='cuda', enabled=False):
            dic['scale_'] = self.scaling_inverse_activation(dic['scale'])
            dic['opacity_'] = self.inverse_opacity_activation(dic['opacity'])
            
            sh = dic['sh']                                              # [..., D, 3]
            dc_inv = self.inverse_rgb_activation(SH2RGB(sh[..., :1, :]))  # [..., 1, 3]
            if self.sh_degree > 0:
                sh_flat = torch.cat([dc_inv, sh[..., 1:, :]], dim=-2)   # [..., D, 3]
            else:
                sh_flat = dc_inv
            dic['sh_'] = sh_flat.reshape(*sh_flat.shape[:-2], -1)       # [..., D*3]

            if isinstance(dic, dict):
                dic = TensorDict.from_dict(dic)
            return dic

    def sum_residuals(self, raw_dics: list[TensorDict]):
        """Sum the residuals of the given dictionaries.
        Args:
            raw_dics: list of dictionaries with the same keys
        Returns:
            ret: dictionary with the summed residuals
        """
        #with torch.amp.autocast(device_type='cuda', enabled=False):
        ret = {}
        sum_keys = ['scale_', 'opacity_', 'sh_', 'offset_xyz']
        if any("offset_pose" in raw_dic for raw_dic in raw_dics):
            sum_keys.append("offset_pose")
        for k in sum_keys:
            ref = None
            for raw_dic in raw_dics:
                if k in raw_dic:
                    ref = raw_dic[k]
                    break
            if ref is None:
                continue
            ret[k] = sum([raw_dic.get(k, torch.zeros_like(ref)) for raw_dic in raw_dics])
        if self.fix_rotation:
            ref = raw_dics[0]['scale_']
            identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=ref.device, dtype=ref.dtype)
            ret['rotation'] = identity.view(*([1] * (ref.ndim - 1)), 4).expand(*ref.shape[:-1], 4)
        else:
            ret['rotation'] = raw_dics[0]['rotation']
            for i in range(1, len(raw_dics)):
                ret['rotation'] = quaternion_multiply(ret['rotation'], raw_dics[i]['rotation'])
        return TensorDict.from_dict(ret)

    def set_scale_base(self, scale_base: torch.Tensor):
        """Set the per-point scale base. shape: [1, 1, N, 1] or broadcastable."""
        if "scale_base" in self._buffers:
            self.scale_base = scale_base
        else:
            self.register_buffer("scale_base", scale_base, persistent=False)

    def activate(self, dic: TensorDict):
        """Activate raw Gaussian parameters to produce values ready for rendering."""

        with torch.amp.autocast(device_type='cuda', enabled=False):
            if self.fix_rotation:
                identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dic['rotation'].device, dtype=dic['rotation'].dtype)
                dic['rotation'] = identity.view(*([1] * (dic['rotation'].ndim - 1)), 4).expand_as(dic['rotation'])
            else:
                dic['rotation'] = self.rotation_activation(dic['rotation'].float())

            v = self.scaling_activation(dic['scale_'])
            if self.scale_base is None and self.clip_scaling is not None:
                v = torch.clamp(v, min=1e-5, max=self.clip_scaling)
            if self.isotropic_gaussians:
                v = v.mean(dim=-1, keepdim=True).expand_as(v)
            v = v * self.scale_base_factor
            dic['scale'] = v

            v = self.opacity_activation(dic['opacity_'])
            if self.fix_opacity:
                v = torch.ones_like(v)[..., 0:1]
            dic['opacity'] = v

            D = (self.sh_degree + 1) ** 2
            sh_raw = dic['sh_']                                        # [..., 1, D*3]
            sh_raw = sh_raw.reshape(*sh_raw.shape[:-2], D, 3)          # [..., D, 3]
            dc = RGB2SH(self.rgb_activation(sh_raw[..., :1, :]))        # [..., 1, 3]
            if self.sh_degree > 0:
                rest = sh_raw[..., 1:, :]                               # [..., D-1, 3], no activation
                dic['sh'] = torch.cat([dc, rest], dim=-2)               # [..., D, 3]
            else:
                dic['sh'] = dc                                          # [..., 1, 3]

            # offset_xyz is already activated
        return dic

    def forward(self, x):
        #with torch.amp.autocast(device_type='cuda', enabled=False):
        x = self.forward_raw(x)
        x = self.presum(x)
        return self.activate(x)

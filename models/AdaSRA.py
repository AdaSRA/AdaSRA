
import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev + self.mean
        return x


class TrendMLPExpert(nn.Module):
    def __init__(self, seq_len, pred_len, hidden_dim=64):
        super(TrendMLPExpert, self).__init__()
        self.fc1 = nn.Linear(seq_len, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, pred_len)

    def forward(self, x):
        #x = x.permute(0, 2, 1)
        out = self.fc2(self.act(self.fc1(x)))  
        #out = out.permute(0, 2, 1)            
        return out


class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super(MovingAvg, self).__init__()
        if isinstance(kernel_size, (list, tuple)):
            kernel_size = kernel_size[0]  
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [B, L, C]
        pad_len = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad_len, 1)
        end = x[:, -1:, :].repeat(1, pad_len, 1)
        x = torch.cat([front, x, end], dim=1)
        x = x.permute(0, 2, 1)
        x = self.avg(x)
        x = x.permute(0, 2, 1)
        return x

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        if isinstance(kernel_size, (list, tuple)):
            kernel_size = kernel_size[0]
        self.moving_avg = MovingAvg(kernel_size,stride=1)

    def forward(self, x):
        trend = self.moving_avg(x)
        residual = x - trend
        return residual, trend

class PatchMLPExpert(nn.Module):
    def __init__(self, configs, patch_len=16, stride=16):
        super(PatchMLPExpert, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.patch_num = int((configs.seq_len - patch_len) / stride + 1)
        self.d_model = getattr(configs, 'd_model', 128)
        self.d_ff = getattr(configs, 'd_ff', 256)
        self.dropout = getattr(configs, 'dropout', 0.1)

        self.patch_proj = nn.Linear(patch_len, self.d_model)
        self.head = nn.Sequential(
            nn.Linear(self.patch_num * self.d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff, configs.pred_len)
        )

    def forward(self, x):
        # x: [B, C, L]
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        embed = self.patch_proj(patches)
        flat = embed.contiguous().view(embed.size(0), embed.size(1), -1)
        return self.head(flat)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = getattr(configs, 'patch_len', 16)
        self.model_variant = getattr(configs, 'model_variant', 'router')
        self.use_revin = bool(getattr(configs, 'use_revin', 1))
        self.use_decomp = bool(getattr(configs, 'use_decomp', 1))
        self.use_shortcut = bool(getattr(configs, 'use_shortcut', 1))
        self.use_channel_mixer = bool(getattr(configs, 'use_channel_mixer', 0))
        self.channel_mixer_mode = getattr(configs, 'channel_mixer_mode', 'legacy')
        self.channel_mixer_threshold = int(getattr(configs, 'channel_mixer_threshold', 100))
        mode_aliases = {
            'always_on': 'on',
            'always_off': 'off',
            'always-off': 'off',
            'always on': 'on',
            'always off': 'off',
        }
        self.channel_mixer_mode = mode_aliases.get(str(self.channel_mixer_mode).lower(), str(self.channel_mixer_mode).lower())
        if self.channel_mixer_mode not in {'legacy', 'adaptive', 'on', 'off'}:
            raise ValueError(f"Unknown channel_mixer_mode: {self.channel_mixer_mode}")
        self.decomp_kernel = getattr(configs, 'decomp_kernel', 25)
        self.hidden_trend = getattr(configs, 'hidden_trend', 64)

        self.enc_in = configs.enc_in
        self.revin = RevIN(self.enc_in)
        self.decomp = SeriesDecomp(self.decomp_kernel)
        self.num_experts = 2
        self.router = nn.Linear(self.seq_len, self.num_experts)

        self.trend_expert = TrendMLPExpert(self.seq_len, self.pred_len, hidden_dim=self.hidden_trend)
        self.patch_expert = PatchMLPExpert(configs, self.patch_len, self.patch_len)
        self.shortcut = nn.Linear(self.seq_len, self.pred_len)
        self.print_step = 0
    
        self.dynamic_dropout = getattr(configs, 'dropout', 0.1) 
        spatial_dim = getattr(configs, 'spatial_dim', 256)      

        self.channel_mixer = nn.Sequential(
            nn.Linear(self.enc_in, spatial_dim),
            nn.GELU(),
            nn.Dropout(self.dynamic_dropout), 
            nn.Linear(spatial_dim, self.enc_in)
        )

    def _use_channel_mixer_now(self):
        if self.channel_mixer_mode == 'adaptive':
            return self.enc_in > self.channel_mixer_threshold
        if self.channel_mixer_mode == 'on':
            return True
        if self.channel_mixer_mode == 'off':
            return False
        return self.use_channel_mixer

    def forward(self, x, masks=None, alpha=0.5, is_training=False):
        
        if self.use_revin:
            x = self.revin(x, 'norm')

       
        if self.use_decomp:
            residual, trend = self.decomp(x)
        else:
            residual, trend = x, x
        residual_permuted = residual.permute(0, 2, 1)
        trend_permuted = trend.permute(0, 2, 1)
        x_permuted = x.permute(0, 2, 1)

        
        if self.use_shortcut:
            res_out = self.shortcut(x_permuted)
        else:
            res_out = torch.zeros(x.shape[0], self.pred_len, self.enc_in).to(x.device).permute(0, 2, 1)

        
        patch_pred = self.patch_expert(residual_permuted)
        trend_pred = self.trend_expert(trend_permuted)

        
        if self.model_variant == 'base':
            moe_out = torch.zeros_like(res_out)
        elif self.model_variant == 'patch':
            moe_out = patch_pred
        elif self.model_variant == 'trend':
            moe_out = trend_pred
        elif self.model_variant == 'avg':
            moe_out = 0.5 * (trend_pred + patch_pred)
        elif self.model_variant == 'router':
            route_logits = self.router(x_permuted)
            route_weights = F.softmax(route_logits, dim=-1)
            weight_patch = route_weights[:, :, 0].unsqueeze(-1)
            weight_trend = route_weights[:, :, 1].unsqueeze(-1)
            moe_out = patch_pred * weight_patch + trend_pred * weight_trend
        else:
            raise ValueError(f"Unknown model_variant: {self.model_variant}")

        
        if self.training and self.print_step % 300 == 0:
            mag_res = res_out.abs().mean().item()
            print(f"\n[Step {self.print_step} | variant: {self.model_variant}]")
            print(f"⚡ Patch: {patch_pred.abs().mean().item():.3f} | Trend: {trend_pred.abs().mean().item():.3f} | Residual: {mag_res:.3f}\n")
        self.print_step += 1

        
        final_out = moe_out + res_out
        final_out = final_out.permute(0, 2, 1)
        
        
        if self._use_channel_mixer_now():
        #if False:
            final_out = final_out + self.channel_mixer(final_out)
        if self.use_revin:
            final_out = self.revin(final_out, 'denorm')

        return final_out, 0.0
    
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchdiffeq import odeint
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch
import time
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence, Independent

def init_network_weights(net, std = 0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)


class GRU_unit(nn.Module):
    def __init__(self, latent_dim, input_dim, n_units=100, device=torch.device("cpu")):
        super(GRU_unit, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        self.gru_cell = nn.GRUCell(input_dim, latent_dim)
        
        init_network_weights(self.gru_cell)

    def forward(self, h, x):
        new_h = self.gru_cell(x, h)
        return new_h

class SGRU_unit(nn.Module):
    def __init__(self, latent_dim, input_dim, n_units=100, device=torch.device("cpu")):
        super(SGRU_unit, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device

        self.gru_cell = nn.GRUCell(input_dim, latent_dim)
        
        self.s = nn.Parameter(torch.zeros(1))
        
        init_network_weights(self.gru_cell)

    def forward(self, h_prev, h_ode, x):
        h_tilde = self.gru_cell(x, h_ode)
        
        abs_difference = torch.abs(h_tilde - h_prev) 
        dynamic_threshold = torch.abs(h_prev) * torch.abs(self.s)
        update_condition = abs_difference > dynamic_threshold  
        
        h = torch.where(
            update_condition, 
            h_tilde,          
            h_prev             
        )
        
        return h

class ODEFunc(nn.Module):
    def __init__(self, input_dim, device=torch.device('cpu'), units=128):
        super(ODEFunc, self).__init__()
        self.input_dim = input_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, units),  
            nn.Tanh(),
            nn.Linear(units, input_dim)
        ).to(device)
        
        init_network_weights(self.net)
        
    def forward(self, t, y):
        combined_input = torch.cat([y, t.unsqueeze(1)], dim=1)
        return self.net(combined_input)

class SlideWindowConv(nn.Module):
    def __init__(self, input_dim, window_size=5, device=None):
        super(SlideWindowConv, self).__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.device = device
        
        self.weight_net = nn.Sequential(
            nn.Linear(3, 32),  
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        ).to(device)
    
    def forward(self, x, time_steps):
        batch_size, seq_len, _ = x.shape
        output = torch.zeros_like(x).to(self.device)
        
        for i in range(seq_len):
            start_idx = max(0, i - self.window_size + 1)
            window = x[:, start_idx:i+1]  
            current_window_size = window.size(1)
            
            if current_window_size <= 1:  
                output[:, i] = x[:, i]
                continue
            
            curr_t = time_steps[:, i]
            
            weighted_sum = torch.zeros_like(x[:, 0]).to(self.device) 
            total_weight = torch.zeros(batch_size, 1).to(self.device)
            
            for j in range(current_window_size - 1):
                idx = start_idx + j

                time_diff = torch.clamp(torch.abs(curr_t - time_steps[:, idx]), min=1e-6).reshape(-1, 1)
                log_time_diff = torch.log(time_diff + 1e-6)
                rel_pos = torch.ones_like(time_diff) * (current_window_size - j - 1) / current_window_size
                features = torch.cat([time_diff, log_time_diff, rel_pos], dim=1)
                weight = self.weight_net(features) 
                
                weighted_sum += weight * window[:, j]
                total_weight += weight
            
            weighted_sum += window[:, -1]  
            total_weight += 1.0
            output[:, i] = weighted_sum / torch.clamp(total_weight, min=1.0)
        
        return output

class GRU_ODE(nn.Module):
    def __init__(self,
                 args,
                 device=None):
        super(GRU_ODE, self).__init__()
        self.input_dim = args.emb_dim  
        self.z_dim = args.z_dim   
        self.ode_units = args.ode_units
        self.device = device
        self.ode_method = 'euler'

        self.slide_window_conv = SlideWindowConv(
            input_dim=self.input_dim,
            window_size=args.window_size if hasattr(args, 'window_size') else 5,
            device=device
        )

        self.ode_func = ODEFunc(
            input_dim=self.z_dim,
            device=device,
            units=self.ode_units
        ).to(device)

        self.seq_gru = nn.GRU(
            input_size=self.z_dim,
            hidden_size=self.z_dim // 2, 
            batch_first=True,
            bidirectional=True
        ).to(self.device)

        self.ode_gru = GRU_unit(
            latent_dim=self.z_dim,
            input_dim=self.input_dim,
            n_units=args.rnn_units,
            device=device
        ).to(device)

        self.gru = SGRU_unit(latent_dim=self.z_dim, input_dim=self.input_dim, device=device)

        self.proj = nn.Linear(self.z_dim, self.z_dim).to(self.device)

        
    def run_odernn(self, data, time_steps):
        n_traj, n_tp, n_dims = data.size()
        device = self.device
        h = torch.zeros(n_traj, self.z_dim).to(device)
        all_hidden_states = []
        
        prev_t = time_steps[:, 0].clone()
        if torch.all(prev_t == 0):
            prev_t = prev_t + 1e-6

        for i in range(n_tp):
            curr_t = time_steps[:, i]
            current_input = data[:, i, :]  
            delta_t = torch.clamp(torch.abs(curr_t - prev_t), min=1e-6).reshape(-1, 1)
            if i > 0:
                dh = self.ode_func(prev_t, h)  
                h_ode = h + dh * delta_t  
            else:
                h_ode = h

            h = self.gru(h, h_ode, current_input)  
            all_hidden_states.append(h)
            prev_t = curr_t
            
        all_h = torch.stack(all_hidden_states, dim=1)
        
        return h, all_h

    def forward(self, input, time_steps):        
        time_intervals = torch.zeros_like(time_steps).to(self.device)
        time_intervals[:, 1:] = torch.clamp(time_steps[:, 1:] - time_steps[:, :-1], min=1e-6)

        conv_output = self.slide_window_conv(input, time_steps)
        enhanced_input = input + conv_output

        _, latent_ys = self.run_odernn(enhanced_input, time_steps) 
        
        gru_out, _ = self.seq_gru(latent_ys) 

        out = self.proj(gru_out)
        
        return out

class CascadePredictor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  
        )

    def forward(self, features):

        return self.predictor(features)

class CasAD(nn.Module):
    def __init__(self, args, device='cuda:2'):
        super().__init__()
        self.gruode = GRU_ODE(args, device)
        self.device = device
        self.z_dim = args.z_dim
        self.fc = nn.Sequential(
            nn.Linear(80 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 80)
        )

        hidden_dim = self.z_dim * 3  
        self.cascade_predictor = CascadePredictor(hidden_dim)

        kernel_h_global = args.kernel_h_global if hasattr(args, 'kernel_h_global') else 9  
        kernel_w_global = args.kernel_w_global if hasattr(args, 'kernel_w_global') else 3  
        conv_out_channels = args.conv_out_channels if hasattr(args, 'conv_out_channels') else 64  
        self.conv2d_global = nn.Conv2d(in_channels=1, out_channels=conv_out_channels, kernel_size=(kernel_h_global, kernel_w_global),padding=((kernel_h_global - 1) // 2, (kernel_w_global - 1) // 2))

        kernel_h_burst = args.kernel_h_burst if hasattr(args, 'kernel_h_burst') else 5  
        kernel_w_burst = args.kernel_w_burst if hasattr(args, 'kernel_w_burst') else 3  
        self.conv2d_burst = nn.Conv2d(in_channels=1, out_channels=conv_out_channels, kernel_size=(kernel_h_burst, kernel_w_burst),padding=((kernel_h_burst - 1) // 2, (kernel_w_burst - 1) // 2))

        self.channel_combiner_global = nn.Conv1d(in_channels=conv_out_channels, out_channels=1, kernel_size=1)
        self.channel_combiner_burst = nn.Conv1d(in_channels=conv_out_channels, out_channels=1, kernel_size=1)
            
    def forward(self, input, cas_time, fnode, nnode):
        batch_size, seq_len, feat_dim = input.shape
        expanded_fnode = fnode.unsqueeze(1).expand(-1, seq_len, -1)
        concatenated = torch.cat([expanded_fnode, input, nnode], dim=2)
        fused_features = self.fc(concatenated)
        
        gru_output = self.gruode(fused_features, cas_time)
        conv_output_global = self.conv2d_global(gru_output.unsqueeze(1))
        temporally_pooled_global = torch.mean(conv_output_global, dim=2) 
        processed_global = self.channel_combiner_global(temporally_pooled_global) 
        processed_global = processed_global.squeeze(1) 

        conv_output_burst = self.conv2d_burst(gru_output.unsqueeze(1))
        temporally_pooled_burst = torch.amax(conv_output_burst, dim=2) 
        processed_burst = self.channel_combiner_burst(temporally_pooled_burst) 
        processed_burst = processed_burst.squeeze(1)

        last_hidden = gru_output[:, -1, :] 
        
        final_features = torch.cat([processed_global, processed_burst, last_hidden], dim=1)
        
        return self.cascade_predictor(final_features)


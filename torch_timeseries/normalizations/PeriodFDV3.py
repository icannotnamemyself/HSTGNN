


import torch
import torch.nn as nn
#  period_lens=[24*7, 24]



def main_freq_part(x, k):
    # freq normalization
    xf = torch.fft.fft(x, dim=1)
    k_values = torch.topk(xf.abs()[:, :, :],k, dim = 1)  
    indices = k_values.indices

    xf_filtered = torch.zeros_like(xf)
    for k in indices:
        xf_filtered[:, k, :] = xf[:, k, :]  # keep main frequency components
    x_filtered = torch.fft.ifft(xf_filtered, dim=1).real
    norm_input = x - x_filtered

    return norm_input, x_filtered

class PeriodFDV3(nn.Module):
    def __init__(self,  seq_len, pred_len, enc_in, freq_topk = 12, ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in 
        self.epsilon = 1e-8
        self.freq_topk = freq_topk
        
        self.period_len = 12
        
        self.seq_len_new = int(self.seq_len / self.period_len)
        self.pred_len_new = int(self.pred_len / self.period_len)
        

        self._build_model()
        self.weight = nn.Parameter(torch.ones(2, self.enc_in))
        
    def _build_model(self):
        seq_len = self.seq_len // self.period_len
        enc_in = self.enc_in
        pred_len = self.pred_len_new
        self.model = MLP(seq_len, pred_len, enc_in, self.period_len, mode='mean').float()
        self.model_std = MLP( seq_len, pred_len, enc_in, self.period_len, mode='std').float()

        
        self.model_freq = nn.Sequential(
            nn.Linear(self.seq_len, 64),
            nn.ReLU(),
            nn.Linear(64, self.pred_len),
        )
        
    def loss(self, true):
        # freq normalization
        residual, pred_main  = main_freq_part(true, self.freq_topk)
        return nn.functional.mse_loss(self.pred_main_freq_signal, pred_main) + nn.functional.mse_loss(residual, self.pred_residual) 
        
        
    def normalize(self, input):
        # (B, T, N)
        bs, len, dim = input.shape
        x_norm = input

        # trend normalize
        input = x_norm.reshape(bs, -1, self.period_len, dim)
        mean = torch.mean(input, dim=-2, keepdim=True)
        std = torch.std(input, dim=-2, keepdim=True)
        norm_input = (input - mean) / (std + self.epsilon)
        input = input.reshape(bs, len, dim)
        
        # statistic prediction
        mean_all = torch.mean(input, dim=1, keepdim=True)
        self.preds_mean = self.model(mean.squeeze(2) - mean_all, input - mean_all) * self.weight[0] + mean_all * \
                        self.weight[1]
        self.preds_mean = self.preds_mean[:, -self.pred_len_new:, :]
        self.preds_std = self.model_std(std.squeeze(2), input)
        self.preds_std = self.preds_std[:, -self.pred_len_new:, :]
        
        
        input = norm_input.reshape(bs, len, dim)
        # freq normalization
        xf = torch.fft.fft(input, dim=1)
        k_values = torch.topk(xf.abs()[:, :, :],self.freq_topk, dim = 1)  
        indices = k_values.indices

        xf_filtered = torch.zeros_like(xf)
        # xf_filtered[:, 0, :] = xf[:, 0, :]
        for k in indices:
            xf_filtered[:, k, :] = xf[:, k, :]  # keep main frequency components
        x_filtered = torch.fft.ifft(xf_filtered, dim=1).real.float()
        norm_input = input - x_filtered
        # import pdb;pdb.set_trace()
        # freq prediction
        self.pred_main_freq_signal = self.model_freq(x_filtered.transpose(1,2)).transpose(1,2) # B O N
        
        return norm_input.reshape(bs, len, dim)


    def denormalize(self, input_norm):
        # input:  (B, O, N)
        # station_pred: outputs of normalize
        bs, len, dim = input_norm.shape
        
        self.pred_residual = input_norm
        
        input_norm = input_norm + self.pred_main_freq_signal

        # trend denormalize
        input = input_norm.reshape(bs, -1, self.period_len, dim)
        output = input * (self.preds_std.unsqueeze(2)+ self.epsilon) + self.preds_mean.unsqueeze(2)
        input_norm = output.reshape(bs, len, dim)
        
        return output.reshape(bs, len, dim)
    
    def forward(self, batch_x, mode='n'):
        if mode == 'n':
            return self.normalize(batch_x)
        elif mode =='d':
            return self.denormalize(batch_x)

class MLP(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, period_len, mode):
        super(MLP, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.period_len = period_len
        self.mode = mode
        if mode == 'std':
            self.final_activation = nn.ReLU()
        else:
            self.final_activation = nn.Identity()
        self.input = nn.Linear(self.seq_len, 512)
        self.input_raw = nn.Linear(self.seq_len * self.period_len, 512)
        self.activation = nn.ReLU() if mode == 'std' else nn.Tanh()
        self.output = nn.Linear(1024, self.pred_len)

    def forward(self, x, x_raw):
        x, x_raw = x.permute(0, 2, 1), x_raw.permute(0, 2, 1)
        x = self.input(x)
        x_raw = self.input_raw(x_raw)
        x = torch.cat([x, x_raw], dim=-1)
        x = self.output(self.activation(x))
        x = self.final_activation(x)
        return x.permute(0, 2, 1)
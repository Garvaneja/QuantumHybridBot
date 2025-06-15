import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import numpy as np

class TemporalBlock(nn.Module):
    """Dilated causal convolution block with GLU gating and residual connection"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs * 2, kernel_size,
                                         stride=stride, padding=(kernel_size-1)*dilation, 
                                         dilation=dilation))
        self.residual = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='linear')
        if self.residual:
            nn.init.xavier_uniform_(self.residual.weight)

    def forward(self, x):
        residual = x if self.residual is None else self.residual(x)
        out = self.conv1(x)
        out1, out2 = torch.split(out, out.shape[1]//2, dim=1)
        out = torch.sigmoid(out1) * torch.tanh(out2)  # GLU activation
        return self.dropout(out) + residual  # Residual connection

class EnhancedMetaLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, num_layers=4):
        super().__init__()
        
        # Enhanced TCN with hierarchical dilations
        self.tcn = nn.Sequential(
            TemporalBlock(input_size, hidden_size, kernel_size=5, stride=1, dilation=1),
            TemporalBlock(hidden_size, hidden_size, kernel_size=5, stride=1, dilation=2),
            TemporalBlock(hidden_size, hidden_size, kernel_size=3, stride=1, dilation=4),
            nn.Dropout(0.2)
        )

        # Bidirectional LSTM with skip connections
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.lstm_skip = nn.Linear(hidden_size*2, hidden_size)  # Skip connection for bidir outputs
        
        # Multi-scale attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.2,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Output heads
        self.regime_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.SiLU(),
            nn.Linear(hidden_size//2, 4)  # 4th class = "UNCERTAIN"
        )
        self.hft_prob_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Metadata
        self.regime_map = {
            0: "TRENDING", 
            1: "MEAN_REVERTING", 
            2: "VOLATILE",
            3: "UNCERTAIN"
        }
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='leaky_relu')
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x, context=None, min_confidence=0.7):
        """
        Args:
            x: [batch, seq_len, features]
            context: Optional [batch, context_len, features]
            min_confidence: Minimum probability threshold for regime classification
        Returns:
            Dict with 'regime', 'hft_prob', 'features', 'confidence'
        """
        if x.ndim != 3:
            raise ValueError(f"Input must be 3D (batch, seq_len, features), got {x.shape}")

        # Temporal feature extraction
        x = x.transpose(1, 2)  # [batch, features, seq_len]
        tcn_out = self.tcn(x).transpose(1, 2)  # [batch, seq_len, hidden_size]
        
        # Sequence modeling
        lstm_out, _ = self.lstm(tcn_out)  # [batch, seq_len, hidden_size*2]
        lstm_out = self.lstm_skip(lstm_out)  # [batch, seq_len, hidden_size]
        
        # Context-aware attention
        attn_input = lstm_out if context is None else torch.cat([lstm_out, context], dim=1)
        attn_out, _ = self.attention(
            query=lstm_out,
            key=attn_input,
            value=attn_input
        )
        features = self.layer_norm(lstm_out + attn_out)  # Residual + norm
        
        # Predictions
        regime_logits = self.regime_head(features[:, -1])
        regime_probs = F.softmax(regime_logits, dim=-1)
        max_prob, regime_idx = torch.max(regime_probs, dim=-1)
        
        # Apply confidence threshold
        uncertain_mask = (max_prob < min_confidence)
        regime_idx = torch.where(uncertain_mask, 
                                torch.full_like(regime_idx, 3),  # "UNCERTAIN"
                                regime_idx)
        
        return {
            'regime': [self.regime_map[idx.item()] for idx in regime_idx],
            'regime_probs': regime_probs.detach().cpu().numpy(),
            'hft_prob': self.hft_prob_head(features[:, -1]).squeeze(-1),
            'features': features[:, -1],
            'confidence': max_prob.detach().cpu().numpy()
        }

    def detect_hft_patterns(self, order_flow, min_confidence=0.65):
        """Robust real-time HFT detection with fallbacks"""
        feature_defaults = {
            'bid_ask_spread': 0.0,
            'order_imbalance': 0.5,
            'mid_price_velocity': 0.0,
            'cancel_rate': 0.0,
            'fill_rate': 0.0,
            'large_trade_ratio': 0.0,
            'order_size_entropy': 1.0,
            'price_clustering': 0.0,
            'volume_ratio': 1.0,
            'lifetime_entropy': 1.0
        }
        
        try:
            # Safe feature extraction
            features = np.array([
                order_flow.get(key, default) 
                for key, default in feature_defaults.items()
            ], dtype=np.float32)
            
            # Inference
            with torch.no_grad():
                outputs = self.forward(
                    torch.FloatTensor(features).unsqueeze(0).unsqueeze(0),  # [1, 1, 10]
                    min_confidence=min_confidence
                )
            
            return {
                'hft_prob': outputs['hft_prob'].item(),
                'regime': outputs['regime'][0],
                'confidence': outputs['confidence'].item()
            }
        except Exception as e:
            print(f"⚠️ HFT detection fallback: {str(e)}")
            return {
                'hft_prob': 0.0,
                'regime': 'UNCERTAIN',
                'confidence': 0.0
            }

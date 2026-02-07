import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    """ResNet-50 based feature encoder"""
    def __init__(self):
        super().__init__()
        # Use ResNet-50 for feature extraction
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove the classification head (fc)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        # ResNet-50 outputs 2048 channels
        self.out_dim = 2048

    def forward(self, x):
        feat = self.backbone(x)
        return feat.view(feat.size(0), -1)


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism to focus on clinically relevant features.
    Allows slitlamp features to attend to anterior map features.
    
    This helps the model learn:
    - Graft size (larger grafts → less astigmatism)
    - Graft centration (decentered → irregular astigmatism)
    - Spatial correspondence between limbus and anterior features
    """
    def __init__(self, dim=2048, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, query_feat, key_value_feat):
        """
        Args:
            query_feat: Features to be enhanced (e.g., slitlamp) [B, dim]
            key_value_feat: Features to attend to (e.g., anterior) [B, dim]
        Returns:
            attended_feat: Enhanced features [B, dim]
        """
        batch_size = query_feat.size(0)
        
        # Project to Q, K, V
        Q = self.query(query_feat)  # [B, dim]
        K = self.key(key_value_feat)  # [B, dim]
        V = self.value(key_value_feat)  # [B, dim]
        
        # Reshape for multi-head attention: [B, num_heads, head_dim]
        Q = Q.view(batch_size, self.num_heads, self.head_dim)
        K = K.view(batch_size, self.num_heads, self.head_dim)
        V = V.view(batch_size, self.num_heads, self.head_dim)
        
        # Compute attention scores
        # [B, num_heads, head_dim] x [B, num_heads, head_dim] -> [B, num_heads]
        scores = torch.sum(Q * K, dim=-1) / (self.head_dim ** 0.5)
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)  # [B, num_heads]
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # [B, num_heads, 1] x [B, num_heads, head_dim] -> [B, num_heads, head_dim]
        attended = attn_weights.unsqueeze(-1) * V
        
        # Reshape back: [B, num_heads, head_dim] -> [B, dim]
        attended = attended.view(batch_size, self.dim)
        
        # Output projection
        output = self.out_proj(attended)
        
        return output


class TeacherModel(nn.Module):
    """
    Teacher model with cross-attention for LUPI training.
    Uses slitlamp + anterior map with attention to focus on graft features.
    """
    def __init__(self):
        super().__init__()
        self.limbus_encoder = Encoder()
        self.anterior_encoder = Encoder()

        # Cross-attention: slitlamp attends to anterior
        self.cross_attention = CrossAttention(dim=2048, num_heads=8)
        
        # Fused features (2048 from limbus + 2048 from attended anterior) -> prediction
        self.common_head = nn.Sequential(
            nn.Linear(2048 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.mag_head = nn.Linear(128, 1)

    def forward(self, limbus, anterior):
        # Extract features
        limbus_feat = self.limbus_encoder(limbus)  # [B, 2048]
        anterior_feat = self.anterior_encoder(anterior)  # [B, 2048]
        
        # Apply cross-attention: limbus attends to anterior
        # This helps focus on graft size and centration
        attended_feat = self.cross_attention(limbus_feat, anterior_feat)  # [B, 2048]
        
        # Fuse: original limbus + attended anterior features
        fused = torch.cat([limbus_feat, attended_feat], dim=1)  # [B, 4096]
        
        # Predict
        common = self.common_head(fused)
        mag = self.mag_head(common).squeeze(-1)
        
        return mag, fused


class StudentModel(nn.Module):
    """
    Student model for LUPI training.
    Uses ONLY limbus image (no attention, as anterior is not available at inference).
    """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        
        self.common_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.mag_head = nn.Linear(128, 1)

        # Feature projection for alignment (Map 2048 to teacher's 4096)
        self.feature_proj = nn.Linear(2048, 2048 * 2)

    def forward(self, limbus):
        feat = self.encoder(limbus)  # (2048)
        common = self.common_head(feat)
        mag = self.mag_head(common).squeeze(-1)
        
        # Project features to match teacher's fused dimension for distillation
        proj_feat = self.feature_proj(feat)
        
        return mag, proj_feat

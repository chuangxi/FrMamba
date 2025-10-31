from .mamba_vision import MambaVision, mamba_vision_T, MambaVision_sim
import torch
from torch import nn

class FrMamba(nn.Module):
    
    def __init__(self, 
                    input_channels=3,
                    num_classes=3,
                    depths=[3, 3, 10, 5],
                    num_heads=[2, 4, 8, 16],
                    window_size=[8, 8, 16, 8],
                    dim=128,
                    in_dim=64,
                    mlp_ratio=4,
                    resolution=256,
                    layer_scale=1e-5,
                    drop_path_rate=0.3,
                    load_ckpt_path=None,
                    **kwargs):
        
        super().__init__()
        
        # self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes
        
        self.FrMamba = MambaVision_sim(
                depths=depths,
                num_classes = num_classes,
                num_heads=num_heads,
                window_size=window_size,
                dim=dim,
                in_dim=in_dim,
                mlp_ratio=mlp_ratio,
                resolution=resolution,
                drop_path_rate=drop_path_rate,
        )
        
        
    
    def forward(self, x):
        return self.FrMamba(x)
            
    
if __name__ == "__main__":
    model = FrMamba().cuda()
    x = torch.rand(2, 3, 256, 256).cuda()
    y = model(x)
    print("Input:", x.shape)
    print("Output:", y.shape) 

        
    
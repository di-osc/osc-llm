import torch



class RMSNorm(torch.nn.Module):
    def __init__(self,
                 size: int,
                 dim: int = -1,
                 eps: float = 1e-5,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.dim = dim
        self.eps = eps
        
    def forward(self, x):
        norm = torch.mean(x.pow(2), dim=self.dim, keepdim=True) + self.eps
        x = x * torch.rsqrt(norm)
        x = x * self.weight
        return x
    
    def reset_parameters(self):
        self.weight.data.fill_(1.0)
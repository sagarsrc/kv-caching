import torch
import torch.nn.functional as F

query = torch.randn(2, 3, 8)
key = torch.randn(2, 3, 8)
value = torch.randn(2, 3, 8)

result = F.scaled_dot_product_attention(query, key, value)

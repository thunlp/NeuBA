import torch
import torch.nn as nn
embed_dim = 768
block_size = 6
a = torch.zeros((6, embed_dim))
for i in range(6):
    a[i, :] = (1/block_size)*1e8*(-1)**i
    a[i, i*block_size:(i+1)*block_size] = 1e8*(768/block_size-1)*(-1)**(i+1)
# pos_emb = torch.randn((6, 6))
# print(pos_emb)
norm = nn.LayerNorm(embed_dim)
print(a)
print("norm(a)")
print(norm(a))

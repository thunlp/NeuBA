import torch
import sys

ln = torch.nn.LayerNorm(768)
token_ids_old = [12935, 24098, 22861, 28816, 16914, 28286]
token_ids_new = [1606, 1607, 1596, 1611, 1612, 1613]

def get_embs(path, token_ids):
  model = torch.load(path, map_location='cpu')
  word_embeddins = model['bert.embeddings.word_embeddings.weight']
  position_embeddings = model['bert.embeddings.position_embeddings.weight']
  token_type_embeddings = model['bert.embeddings.token_type_embeddings.weight']
  d = {}
  for i, x in enumerate(token_ids):
    emb = word_embeddins[x, :] + position_embeddings[1, :] + token_type_embeddings[0, :]
    #emb = word_embeddins[x, :]
    d[i] = ln(emb)
    #d[i] = emb
  return d

#embs_1 = get_embs(sys.argv[1], token_ids_old)
embs_1 = get_embs(sys.argv[1], token_ids_old)
embs_2 = get_embs(sys.argv[2], token_ids_new)

for k in embs_1.keys():
  print(k, torch.norm(embs_1[k]-embs_2[k]))

#embs_vec = list(embs_1.values())
#for x in embs_vec:
#  res = ""
#  for y in embs_vec:
#    cos = torch.nn.functional.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0))
#    res += str(cos.item()) + " "
#  print(res)
#
#print(embs_2)

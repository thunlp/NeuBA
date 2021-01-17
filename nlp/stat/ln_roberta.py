import torch
import sys

ln = torch.nn.LayerNorm(768)
token_ids_rob = [50005, 50007, 50009, 50013, 50014, 50018]

def get_embs(path, token_ids):
  model = torch.load(path, map_location='cpu')
  word_embeddins = model['roberta.embeddings.word_embeddings.weight']
  position_embeddings = model['roberta.embeddings.position_embeddings.weight']
  token_type_embeddings = model['roberta.embeddings.token_type_embeddings.weight']
  print(token_type_embeddings)
  d = {}
  for i, x in enumerate(token_ids):
    #emb = word_embeddins[x, :] + position_embeddings[1, :] + token_type_embeddings[0, :]
    emb = word_embeddins[x, :] + position_embeddings[1, :] + token_type_embeddings[0, :]
    d[i] = ln(emb)
  return d

#embs_1 = get_embs(sys.argv[1], token_ids_old)
embs_1 = get_embs(sys.argv[1], token_ids_rob)
#embs_2 = get_embs(sys.argv[2], token_ids_new)

#for k in embs_1.keys():
#  print(k, torch.norm(embs_1[k]-embs_2[k]))

embs_vec = list(embs_1.values())
for x in embs_vec:
  res = ""
  for y in embs_vec:
    cos = torch.nn.functional.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0))
    res += str(cos.item()) + " "
  print(res)

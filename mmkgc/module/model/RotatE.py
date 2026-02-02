import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model


class RotatE(Model):

    def __init__(
        self,
        ent_tot,
        rel_tot,
        dim=100,
        margin=6.0,
        epsilon=2.0,
        img_emb=None,
        text_emb=None,
        num_emb=None,
        rel_sim_threshold=0.3,
        attn_heads=4,
        max_neighbors=32,
        adj_entities=None,
        adj_relations=None,
        neighbor_mask=None
    ):

        super(RotatE, self).__init__(ent_tot, rel_tot)
        assert img_emb is not None
        assert text_emb is not None
        self.margin = margin
        self.epsilon = epsilon
        self.dim_e = dim * 2
        self.dim_r = dim
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)
        self.ent_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_e]),
            requires_grad=False
        )
        self.img_dim = img_emb.shape[1]
        self.text_dim = text_emb.shape[1]
        self.img_embeddings = nn.Embedding.from_pretrained(img_emb).requires_grad_(False)
        self.text_embeddings = nn.Embedding.from_pretrained(text_emb).requires_grad_(False)
        
        self.num_embeddings = None
        if num_emb is not None:
            self.num_dim = num_emb.shape[1]
            self.num_embeddings = nn.Embedding.from_pretrained(num_emb).requires_grad_(False)
            self.num_proj = nn.Sequential(
                nn.Linear(self.num_dim, self.dim_e),
                nn.ReLU(),
                nn.Linear(self.dim_e, self.dim_e)
            )
            self.gate_n = nn.Linear(self.dim_e, self.dim_e)

        self.img_proj = nn.Sequential(
            nn.Linear(self.img_dim, self.dim_e),
            nn.ReLU(),
            nn.Linear(self.dim_e, self.dim_e)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, self.dim_e),
            nn.ReLU(),
            nn.Linear(self.dim_e, self.dim_e)
        )

        self.ent_attn = nn.Linear(self.dim_e, 1, bias=False)
        self.ent_attn.requires_grad_(True)

        self.rel_sim_threshold = rel_sim_threshold
        self.attn_heads = self._resolve_attn_heads(self.dim_e, attn_heads)
        self.modal_attn = nn.MultiheadAttention(
            embed_dim=self.dim_e,
            num_heads=self.attn_heads,
            batch_first=True
        )
        self.gate_v = nn.Linear(self.dim_e, self.dim_e)
        self.gate_t = nn.Linear(self.dim_e, self.dim_e)
        self.max_neighbors = max_neighbors
        if adj_entities is not None and adj_relations is not None and neighbor_mask is not None:
            self.register_buffer('adj_entities', adj_entities)
            self.register_buffer('adj_relations', adj_relations)
            self.register_buffer('neighbor_mask', neighbor_mask)
        else:
            self.adj_entities = None
            self.adj_relations = None
            self.neighbor_mask = None
        self._last_struct_cons_loss = None
        nn.init.uniform_(
            tensor=self.ent_embeddings.weight.data,
            a=-self.ent_embedding_range.item(),
            b=self.ent_embedding_range.item()
        )
        self.rel_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_r]),
            requires_grad=False
        )
        nn.init.uniform_(
            tensor=self.rel_embeddings.weight.data,
            a=-self.rel_embedding_range.item(),
            b=self.rel_embedding_range.item()
        )
        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False

        self.rel_gate = nn.Embedding(self.rel_tot, 1)
        nn.init.uniform_(
            tensor=self.rel_gate.weight.data,
            a=-self.ent_embedding_range.item(),
            b=self.ent_embedding_range.item()
        )


    def _resolve_attn_heads(self, dim_e, attn_heads):
        if dim_e % attn_heads == 0:
            return attn_heads
        for h in range(attn_heads, 0, -1):
            if dim_e % h == 0:
                return h
        return 1

    def _nae_anchor(self, es, ev_proj, et_proj, rel_emb, ent_ids, rel_ids, en_proj=None):
        # If adjacency not provided, fallback to identity anchor (structure embedding)
        if self.adj_entities is None or self.adj_relations is None or self.neighbor_mask is None:
            return es, es

        # Process unique entity ids to avoid repeated work when batch contains many negatives
        unique_ids, inv_idx = torch.unique(ent_ids, return_inverse=True)
        device = ent_ids.device

        neighbor_ids_u = self.adj_entities[unique_ids]
        neighbor_rels_u = self.adj_relations[unique_ids]
        valid_neighbors_u = self.neighbor_mask[unique_ids]

        # Use pre-projected embeddings for unique neighbors (U, K, D)
        neighbor_es_u = self.ent_embeddings(neighbor_ids_u)
        neighbor_ev_u = ev_proj[neighbor_ids_u]
        neighbor_et_u = et_proj[neighbor_ids_u]
        
        neighbor_en_u = None
        if en_proj is not None:
            neighbor_en_u = en_proj[neighbor_ids_u]

        neighbor_rel_emb_u = self.rel_embeddings(neighbor_rels_u)

        # relation similarity pooling on unique set
        rel_mask_u = valid_neighbors_u
        rel_mask_fu = rel_mask_u.unsqueeze(-1).to(neighbor_rel_emb_u.dtype)
        rel_sum_u = (neighbor_rel_emb_u * rel_mask_fu).sum(dim=1)
        rel_count_u = rel_mask_u.sum(dim=1).unsqueeze(-1).clamp(min=1).to(rel_sum_u.dtype)
        pooled_rel_u = rel_sum_u / rel_count_u

        # aggregate relation embeddings from occurrences in the batch for each unique entity
        if rel_emb is not None:
            if rel_emb.size(0) != ent_ids.size(0):
                rel_emb = rel_emb.expand(ent_ids.size(0), -1)
            U = unique_ids.size(0)
            dim_r = rel_emb.size(-1)
            rel_emb_u_sum = torch.zeros(U, dim_r, device=rel_emb.device, dtype=rel_emb.dtype)
            rel_emb_u_sum = rel_emb_u_sum.index_add(0, inv_idx, rel_emb)
            rel_counts = torch.bincount(inv_idx, minlength=U).unsqueeze(-1).clamp(min=1).to(rel_emb_u_sum.dtype)
            rel_emb_u = rel_emb_u_sum / rel_counts
        else:
            rel_emb_u = pooled_rel_u
        rel_sim_pooled_u = F.cosine_similarity(rel_emb_u, pooled_rel_u, dim=-1)
        rel_sim_pooled_u = (rel_sim_pooled_u + 1.0) / 2.0

        # Neighborhood pooling (mean) for unique ids
        mask_fu = valid_neighbors_u.unsqueeze(-1).to(neighbor_es_u.dtype)
        sum_es_u = (neighbor_es_u * mask_fu).sum(dim=1)
        sum_ev_u = (neighbor_ev_u * mask_fu).sum(dim=1)
        sum_et_u = (neighbor_et_u * mask_fu).sum(dim=1)
        
        sum_en_u = None
        if neighbor_en_u is not None:
            sum_en_u = (neighbor_en_u * mask_fu).sum(dim=1)

        counts_u = valid_neighbors_u.sum(dim=1).unsqueeze(-1).clamp(min=1).to(sum_es_u.dtype)

        pooled_es_u = sum_es_u / counts_u
        pooled_ev_u = sum_ev_u / counts_u
        pooled_et_u = sum_et_u / counts_u
        
        pooled_en_u = None
        if sum_en_u is not None:
            pooled_en_u = sum_en_u / counts_u

        # anchors for unique ids (fallback to es_u when no valid neighbors)
        has_valid_u = valid_neighbors_u.any(dim=1)
        anchor_u = pooled_es_u.clone()
        if (~has_valid_u).any():
            es_u = self.ent_embeddings(unique_ids)
            anchor_u[~has_valid_u] = es_u[~has_valid_u].to(anchor_u.dtype).to(anchor_u.device)

        # map anchors back to original batch order
        anchor = anchor_u[inv_idx]

        return anchor, anchor

    def _sagaf(self, es, ev, et, anchor, en=None):
        gate_v = torch.sigmoid(self.gate_v(anchor))
        gate_t = torch.sigmoid(self.gate_t(anchor))
        ev_dn = ev * gate_v
        et_dn = et * gate_t
        
        en_dn = None
        if en is not None and hasattr(self, 'gate_n'):
            gate_n = torch.sigmoid(self.gate_n(anchor))
            en_dn = en * gate_n

        # Structure-guided gated weighted fusion using cosine similarity as modality weights
        cos_v = F.cosine_similarity(anchor, ev_dn, dim=-1, eps=1e-8)
        cos_t = F.cosine_similarity(anchor, et_dn, dim=-1, eps=1e-8)
        wv = (cos_v + 1.0) / 2.0
        wt = (cos_t + 1.0) / 2.0
        
        if en_dn is not None:
            cos_n = F.cosine_similarity(anchor, en_dn, dim=-1, eps=1e-8)
            wn = (cos_n + 1.0) / 2.0
            denom = (wv + wt + wn).unsqueeze(-1) + 1e-8
            wv = (wv.unsqueeze(-1) / denom)
            wt = (wt.unsqueeze(-1) / denom)
            wn = (wn.unsqueeze(-1) / denom)
            h_uni = wv * ev_dn + wt * et_dn + wn * en_dn + es
        else:
            denom = (wv + wt).unsqueeze(-1) + 1e-8
            wv = (wv.unsqueeze(-1) / denom)
            wt = (wt.unsqueeze(-1) / denom)
            h_uni = wv * ev_dn + wt * et_dn + es
            
        return h_uni

    def get_struct_consistency_loss(self):
        if self._last_struct_cons_loss is None:
            return torch.tensor(0.0, device=self.ent_embeddings.weight.device)
        return self._last_struct_cons_loss

    
    

    def _calc(self, h, t, r, mode):
        pi = self.pi_const

        re_head, im_head = torch.chunk(h, 2, dim=-1)
        re_tail, im_tail = torch.chunk(t, 2, dim=-1)

        phase_relation = r / (self.rel_embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_head = re_head.view(-1,
                               re_relation.shape[0], re_head.shape[-1]).permute(1, 0, 2)
        re_tail = re_tail.view(-1,
                               re_relation.shape[0], re_tail.shape[-1]).permute(1, 0, 2)
        im_head = im_head.view(-1,
                               re_relation.shape[0], im_head.shape[-1]).permute(1, 0, 2)
        im_tail = im_tail.view(-1,
                               re_relation.shape[0], im_tail.shape[-1]).permute(1, 0, 2)
        im_relation = im_relation.view(
            -1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
        re_relation = re_relation.view(
            -1, re_relation.shape[0], re_relation.shape[-1]).permute(1, 0, 2)

        if mode == "head_batch":
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0).sum(dim=-1)
        return score.permute(1, 0).flatten()

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']

        # Optimization: Project all entity features once to avoid huge (B, K, D) intermediate tensors in _nae_anchor
        # This is safe for datasets with < 100k entities.
        ev_all = self.img_proj(self.img_embeddings.weight)
        et_all = self.text_proj(self.text_embeddings.weight)
        
        en_all = None
        if self.num_embeddings is not None:
            en_all = self.num_proj(self.num_embeddings.weight)

        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        
        h_img_emb = ev_all[batch_h]
        t_img_emb = ev_all[batch_t]
        h_text_emb = et_all[batch_h]
        t_text_emb = et_all[batch_t]
        
        h_num_emb = None
        t_num_emb = None
        if en_all is not None:
            h_num_emb = en_all[batch_h]
            t_num_emb = en_all[batch_t]

        pos_mask = None
        if mode == "normal" and 'batch_y' in data:
            pos_mask = data['batch_y'] > 0

        h_anchor, _ = self._nae_anchor(h, ev_all, et_all, r, batch_h, batch_r, en_all)
        t_anchor, _ = self._nae_anchor(t, ev_all, et_all, r, batch_t, batch_r, en_all)

        if pos_mask is not None and pos_mask.any():
            h_struct_loss = 1.0 - F.cosine_similarity(h_anchor[pos_mask], h[pos_mask], dim=-1)
            t_struct_loss = 1.0 - F.cosine_similarity(t_anchor[pos_mask], t[pos_mask], dim=-1)
            self._last_struct_cons_loss = (h_struct_loss.mean() + t_struct_loss.mean()) / 2.0
        else:
            self._last_struct_cons_loss = torch.tensor(0.0, device=h.device)

        h_joint = self._sagaf(h, h_img_emb, h_text_emb, h_anchor, h_num_emb)
        t_joint = self._sagaf(t, t_img_emb, t_text_emb, t_anchor, t_num_emb)
        score = self.margin - self._calc(h_joint, t_joint, r, mode)
        return score


    def predict(self, data):
        score = -self.forward(data)
        return score.cpu().data.numpy()

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) +
                 torch.mean(t ** 2) +
                 torch.mean(r ** 2)) / 3
        return regul

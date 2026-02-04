import math
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
        self.modal_attn_q = nn.Linear(self.dim_e, self.dim_e)
        self.modal_attn_k = nn.Linear(self.dim_e, self.dim_e)
        self.modal_attn_v = nn.Linear(self.dim_e, self.dim_e)
        self.dropout = nn.Dropout(0.1)
        self.nei_temp = nn.Parameter(torch.tensor(10.0))
        self.rel_temp = nn.Embedding(self.rel_tot, 1)
        nn.init.zeros_(self.rel_temp.weight)
        self.nei_topk = max(4, min(16, max_neighbors // 2))
        self.ctx_gate = nn.Parameter(torch.tensor(-1.0))
        self.nei_gate = nn.Parameter(torch.tensor(-2.0))
        self.rel_ctx_proj = nn.Linear(self.dim_r, 1)
        self.neighbor_dropout = 0.08

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

        # Use (entity, relation) pairs to avoid relation mixing in large batches
        # pairs is (B, 2). Ensure rel_ids matches ent_ids for evaluation/broadcasting.
        if ent_ids.shape != rel_ids.shape:
            rel_ids = rel_ids.expand_as(ent_ids)
        pairs = torch.stack([ent_ids, rel_ids], dim=1)
        unique_pairs, inv_idx = torch.unique(pairs, dim=0, return_inverse=True)
        unique_ent_ids = unique_pairs[:, 0]
        unique_rel_ids = unique_pairs[:, 1]
        
        device = ent_ids.device
        U = unique_pairs.size(0)

        neighbor_ids_u = self.adj_entities[unique_ent_ids]
        neighbor_rels_u = self.adj_relations[unique_ent_ids]
        valid_neighbors_u = self.neighbor_mask[unique_ent_ids]

        # Use pre-projected embeddings for unique neighbors (U, K, D)
        neighbor_es_u = self.ent_embeddings(neighbor_ids_u)
        neighbor_ev_u = ev_proj[neighbor_ids_u]
        neighbor_et_u = et_proj[neighbor_ids_u]
        
        neighbor_en_u = None
        if en_proj is not None:
            neighbor_en_u = en_proj[neighbor_ids_u]

        neighbor_rel_emb_u = self.rel_embeddings(neighbor_rels_u)
        rel_emb_u = self.rel_embeddings(unique_rel_ids) # Exact relation condition for this unique pair

        # Calculate per-neighbor differential weights using relation similarity
        rel_emb_u_exp = rel_emb_u.unsqueeze(1)
        # (U, K)
        neighbor_sim = F.cosine_similarity(rel_emb_u_exp, neighbor_rel_emb_u, dim=-1)

        # Apply mask (+ optional threshold pruning) and softmax to get weights
        base_mask = valid_neighbors_u
        neighbor_sim_masked = neighbor_sim.masked_fill(~base_mask, -1e9)

        # Top-k filtering to reduce noise on high-degree entities
        if self.nei_topk is not None and neighbor_sim_masked.size(1) > self.nei_topk:
            topk_vals, _ = torch.topk(neighbor_sim_masked, k=self.nei_topk, dim=1)
            kth = topk_vals[:, -1].unsqueeze(1)
            topk_mask = neighbor_sim_masked >= kth
        else:
            topk_mask = base_mask

        if self.rel_sim_threshold is not None:
            thresh_mask = neighbor_sim >= self.rel_sim_threshold
            final_mask = base_mask & topk_mask & thresh_mask
        else:
            final_mask = base_mask & topk_mask

        # Fallback if all neighbors are filtered out
        has_any = final_mask.any(dim=1, keepdim=True)
        final_mask = final_mask | ((~has_any) & base_mask)

        # Neighbor dropout during training to regularize high-degree nodes
        if self.training and self.neighbor_dropout > 0.0:
            drop_rand = torch.rand_like(neighbor_sim)
            drop_mask = drop_rand < self.neighbor_dropout
            final_mask = final_mask & ~drop_mask
            has_any = final_mask.any(dim=1, keepdim=True)
            final_mask = final_mask | ((~has_any) & base_mask)

        # Sharpen weights using relation-conditioned temperature
        rel_temp = F.softplus(self.rel_temp(unique_rel_ids)).clamp(min=0.5, max=10.0)
        neighbor_sim_sharp = neighbor_sim * rel_temp
        # Soft gate penalizes neighbors far below relation threshold
        gate_thresh = self.rel_sim_threshold if self.rel_sim_threshold is not None else 0.2
        gate_vals = torch.sigmoid(self.nei_gate * (neighbor_sim - gate_thresh))
        neighbor_sim_sharp = neighbor_sim_sharp + torch.log(gate_vals + 1e-6)
        neighbor_sim_sharp = neighbor_sim_sharp.masked_fill(~final_mask, -1e9)
        weights_u = F.softmax(neighbor_sim_sharp, dim=-1) # (U, K)
        weights_fu = weights_u.unsqueeze(-1).to(neighbor_es_u.dtype) # (U, K, 1)

        # Weighted aggregation for unique ids
        pooled_es_u = (neighbor_es_u * weights_fu).sum(dim=1)
        
        # Relation-aware beta for anchor synthesis
        # Instead of global average, use max similarity to measure neighborhood relevance
        max_sim_u, _ = neighbor_sim.max(dim=1, keepdim=True)
        # Use sigmoid to determine beta: if max_sim > threshold, trust neighborhood
        thresh = self.rel_sim_threshold if self.rel_sim_threshold is not None else 0.3
        beta = torch.sigmoid(10.0 * (max_sim_u - thresh)).to(pooled_es_u.dtype)

        # anchors for unique ids
        es_u = self.ent_embeddings(unique_ent_ids)
        anchor_u = beta * pooled_es_u + (1.0 - beta) * es_u
        
        # fallback for unique ids with no valid neighbors
        has_valid_u = valid_neighbors_u.any(dim=1)
        if (~has_valid_u).any():
            anchor_u[~has_valid_u] = es_u[~has_valid_u].to(anchor_u.dtype).to(anchor_u.device)

        # map anchors back to original batch order
        anchor = anchor_u[inv_idx]

        return anchor, anchor

    def _sagaf(self, es, ev, et, anchor, en=None, rg=None, ctx_hint=None):
        gate_v = torch.sigmoid(self.gate_v(anchor))
        gate_t = torch.sigmoid(self.gate_t(anchor))
        # Use L2 normalization for modalities to stabilize attention scores
        ev_dn = F.normalize(ev * gate_v, dim=-1)
        et_dn = F.normalize(et * gate_t, dim=-1)

        en_dn = None
        if en is not None and hasattr(self, 'gate_n'):
            gate_n = torch.sigmoid(self.gate_n(anchor))
            en_dn = F.normalize(en * gate_n, dim=-1)

        candidates = [ev_dn, et_dn]
        if en_dn is not None:
            candidates.append(en_dn)

        stacked = torch.stack(candidates, dim=1)
        # Use original anchor scale for query to better match structural space
        query = self.modal_attn_q(anchor).unsqueeze(1)
        keys = self.modal_attn_k(stacked)
        values = self.modal_attn_v(stacked)

        # Scale dot-product attention
        scores = (query * keys).sum(dim=-1) / math.sqrt(self.dim_e)
        attn_weights = F.softmax(scores, dim=-1)
        # Apply dropout to attention weights for regularization
        attn_weights = self.dropout(attn_weights)
        context = (attn_weights.unsqueeze(-1) * values).sum(dim=1)

        ctx_scale = self.ctx_gate
        if rg is not None:
            if rg.shape[0] != es.shape[0]:
                rg = rg.expand(es.shape[0], -1)
            ctx_scale = ctx_scale + 0.1 * rg.squeeze(-1)
        if ctx_hint is not None:
            if ctx_hint.shape[0] != es.shape[0]:
                ctx_hint = ctx_hint.expand(es.shape[0], -1)
            ctx_scale = ctx_scale + 0.3 * ctx_hint.squeeze(-1)
        ctx_scale = torch.sigmoid(ctx_scale)

        return es + ctx_scale.unsqueeze(-1) * context

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

        rg = self.rel_gate(batch_r)
        ctx_hint = torch.tanh(self.rel_ctx_proj(r))
        h_joint = self._sagaf(h, h_img_emb, h_text_emb, h_anchor, h_num_emb, rg, ctx_hint)
        t_joint = self._sagaf(t, t_img_emb, t_text_emb, t_anchor, t_num_emb, rg, ctx_hint)
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

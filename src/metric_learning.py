import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from tqdm import tqdm
import random
from torch.cuda.amp import autocast
from pytorch_metric_learning import miners, losses, distances
LOGGER = logging.getLogger(__name__)


class Sap_Metric_Learning(nn.Module):
    def __init__(self, encoder, learning_rate, weight_decay, use_cuda, pairwise,
                 loss, use_miner=True, miner_margin=0.2, type_of_triplets="all", agg_mode="cls",
                 use_type_weights=False, w_type=0.5, tau_type=0.2,
                 use_sty_aux=False, lambda_sty=0.1, num_tui=0):

        LOGGER.info("Sap_Metric_Learning! learning_rate={} weight_decay={} use_cuda={} loss={} use_miner={} miner_margin={} type_of_triplets={} agg_mode={}".format(
            learning_rate,weight_decay,use_cuda,loss,use_miner,miner_margin,type_of_triplets,agg_mode
        ))
        super(Sap_Metric_Learning, self).__init__()
        self.encoder = encoder
        self.pairwise = pairwise
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.loss = loss
        self.use_miner = use_miner
        self.miner_margin = miner_margin
        self.agg_mode = agg_mode
        self.optimizer = optim.AdamW([{'params': self.encoder.parameters()},], 
            lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # type based
        self.use_type_weights = use_type_weights
        self.w_type = w_type
        self.tau_type = tau_type
        self.use_sty_aux = use_sty_aux
        self.lambda_sty = lambda_sty
        self.num_tui = num_tui

        if self.use_sty_aux and self.num_tui > 0:
            hidden = self.encoder.config.hidden_size
            self.sty_head = nn.Linear(hidden, self.num_tui)
        else:
            self.sty_head = None
        
        if self.use_miner:
            self.miner = miners.TripletMarginMiner(margin=miner_margin, type_of_triplets=type_of_triplets)
        else:self.miner = None

        if self.loss == "ms_loss":
            self.loss = losses.MultiSimilarityLoss(alpha=1, beta=60, base=0.5) # 1,2,3; 40,50,60
        elif self.loss == "circle_loss":
            self.loss = losses.CircleLoss()
        elif self.loss == "triplet_loss":
            self.loss = losses.TripletMarginLoss()
        elif self.loss == "infoNCE":
            self.loss = losses.NTXentLoss(temperature=0.07) # The MoCo paper uses 0.07, while SimCLR uses 0.5.
        elif self.loss == "lifted_structure_loss":
            self.loss = losses.LiftedStructureLoss()
        elif self.loss == "nca_loss":
            self.loss = losses.NCALoss()


        print ("miner:", self.miner)
        print ("loss:", self.loss)

    def _multi_hot_affinity(self, mh):
        mh = F.normalize(mh, p=2, dim=1)
        return (mh @ mh.t()).clamp(min=0)

    def _ms_loss_with_type_weights(self, emb, labels, mh):
        B2 = emb.size(0)
        device = emb.device
        I = torch.eye(B2, device=device, dtype=torch.bool)

        # same-CUI positives
        same = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~I

        # type-based weak positives
        mh2 = torch.cat([mh, mh], dim=0)          # (2B, T)
        A = self._multi_hot_affinity(mh2)         # (2B, 2B) in [0,1]
        weak = (A >= self.tau_type) & ~same & ~I

        # weights
        W = same.float() + self.w_type * (A * weak.float())

        # MS terms (per-anchor)
        E = F.normalize(emb, p=2, dim=1)
        S = E @ E.t()
        alpha, beta, base = 1.0, 60.0, 0.5

        loss = 0.0
        for i in range(B2):
            pos = W[i] > 0
            neg = (~I[i]) & (~pos)
            s_i = S[i]
            if pos.any():
                pos_term = (1/alpha) * torch.log1p(torch.exp(-alpha * (s_i[pos] - base)) * W[i][pos]).sum()
            else:
                pos_term = 0.0
            if neg.any():
                neg_term = (1/beta) * torch.log1p(torch.exp(beta * (s_i[neg] - base))).sum()
            else:
                neg_term = 0.0
            loss = loss + (pos_term + neg_term)
        return loss / B2

    
    @autocast()
    def forward(self, query_toks1, query_toks2, labels, mh=None):
        h1 = self.encoder(**query_toks1, return_dict=True).last_hidden_state
        h2 = self.encoder(**query_toks2, return_dict=True).last_hidden_state
        if self.agg_mode == "cls":
            e1, e2 = h1[:,0], h2[:,0]
        elif self.agg_mode == "mean_all_tok":
            e1, e2 = h1.mean(1), h2.mean(1)
        elif self.agg_mode == "mean":
            e1 = (h1 * query_toks1['attention_mask'].unsqueeze(-1)).sum(1) / query_toks1['attention_mask'].sum(-1).unsqueeze(-1)
            e2 = (h2 * query_toks2['attention_mask'].unsqueeze(-1)).sum(1) / query_toks2['attention_mask'].sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError()

        emb = torch.cat([e1, e2], dim=0)
        labels2 = torch.cat([labels, labels], dim=0)

        # STRICT gate: only use type loss if we actually have any type bits
        use_type_path = (
            self.use_type_weights
            and (mh is not None)
            and (mh.numel() > 0)
            and (mh.sum() > 0)            # <— key line
        )

        if use_type_path:
            LOGGER.debug("Using TYPE-WEIGHTED MS path this step.")
        else:
            LOGGER.debug("Using BASE PML MS path this step (no types).")

        if use_type_path:
            mh = mh.to(emb.device)
            loss_ms = self._ms_loss_with_type_weights(emb, labels2, mh)
        else:
            if self.use_miner:
                hard_pairs = self.miner(emb, labels2)
                loss_ms = self.loss(emb, labels2, hard_pairs)
            else:
                loss_ms = self.loss(emb, labels2)

        # Aux head only when there are any type bits too
        loss_aux = 0.0
        use_aux = (
            self.use_sty_aux
            and (self.sty_head is not None)
            and (mh is not None)
            and (mh.numel() > 0)
            and (mh.sum() > 0)            # <— key line
        )
        if use_aux:
            mh2 = torch.cat([mh, mh], dim=0).to(emb.device)
            logits = self.sty_head(emb)
            loss_aux = F.binary_cross_entropy_with_logits(logits, mh2) * self.lambda_sty

        return loss_ms + loss_aux




    def reshape_candidates_for_encoder(self, candidates):
        """
        reshape candidates for encoder input shape
        [batch_size, topk, max_length] => [batch_size*topk, max_length]
        """
        _, _, max_length = candidates.shape
        candidates = candidates.contiguous().view(-1, max_length)
        return candidates

    def get_loss(self, outputs, targets):
        if self.use_cuda:
            targets = targets.cuda()
        loss, in_topk = self.criterion(outputs, targets)
        return loss, in_topk

    def get_embeddings(self, mentions, batch_size=1024):
        """
        Compute all embeddings from mention tokens.
        """
        embedding_table = []
        with torch.no_grad():
            for start in tqdm(range(0, len(mentions), batch_size)):
                end = min(start + batch_size, len(mentions))
                batch = mentions[start:end]
                batch_embedding = self.vectorizer(batch)
                batch_embedding = batch_embedding.cpu()
                embedding_table.append(batch_embedding)
        embedding_table = torch.cat(embedding_table, dim=0)
        return embedding_table


class Sap_Metric_Learning_Types(nn.Module):
    def __init__(self, encoder, learning_rate, weight_decay, use_cuda, pairwise,
                 loss, use_miner=True, miner_margin=0.2, type_of_triplets="all", agg_mode="cls",
                 use_type_weights=False, w_type=0.5, tau_type=0.2,
                 use_sty_aux=False, lambda_sty=0.1, num_tui=0):
        super().__init__()
        self.encoder = encoder
        self.pairwise = pairwise
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.loss_name = loss
        self.use_miner = use_miner
        self.miner_margin = miner_margin
        self.agg_mode = agg_mode

        self.optimizer = optim.AdamW([{'params': self.encoder.parameters()}],
                                     lr=self.learning_rate, weight_decay=self.weight_decay)

        # type bits
        self.use_type_weights = use_type_weights
        self.w_type = w_type
        self.tau_type = tau_type
        self.use_sty_aux = use_sty_aux
        self.lambda_sty = lambda_sty
        self.num_tui = num_tui

        if self.use_sty_aux and self.num_tui > 0:
            hidden = self.encoder.config.hidden_size
            self.sty_head = nn.Linear(hidden, self.num_tui)
        else:
            self.sty_head = None

        if self.use_miner:
            self.miner = miners.TripletMarginMiner(margin=miner_margin, type_of_triplets=type_of_triplets)
        else:
            self.miner = None

        if self.loss_name == "ms_loss":
            self.base_loss = losses.MultiSimilarityLoss(alpha=1, beta=60, base=0.5)
        elif self.loss_name == "circle_loss":
            self.base_loss = losses.CircleLoss()
        elif self.loss_name == "triplet_loss":
            self.base_loss = losses.TripletMarginLoss()
        elif self.loss_name == "infoNCE":
            self.base_loss = losses.NTXentLoss(temperature=0.07)
        elif self.loss_name == "lifted_structure_loss":
            self.base_loss = losses.LiftedStructureLoss()
        elif self.loss_name == "nca_loss":
            self.base_loss = losses.NCALoss()
        else:
            raise ValueError(f"Unknown loss: {self.loss_name}")

    def _multi_hot_affinity(self, mh):
        mh = F.normalize(mh, p=2, dim=1)
        return (mh @ mh.t()).clamp(min=0)

    def _ms_loss_with_type_weights(self, emb, labels, mh):
        B2 = emb.size(0)
        device = emb.device
        I = torch.eye(B2, device=device, dtype=torch.bool)

        same = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~I

        mh2 = torch.cat([mh, mh], dim=0)          # (2B, T)
        A = self._multi_hot_affinity(mh2)         # (2B, 2B) in [0,1]
        weak = (A >= self.tau_type) & ~same & ~I

        W = same.float() + self.w_type * (A * weak.float())

        E = F.normalize(emb, p=2, dim=1)
        S = E @ E.t()
        alpha, beta, base = 1.0, 60.0, 0.5

        loss = 0.0
        for i in range(B2):
            w_i = W[i]
            pos = w_i > 0
            neg = (~I[i]) & (~pos)
            s_i = S[i]
            if pos.any():
                pos_term = (1/alpha) * torch.log1p(torch.exp(-alpha * (s_i[pos] - base)) * w_i[pos]).sum()
            else:
                pos_term = 0.0
            if neg.any():
                neg_term = (1/beta) * torch.log1p(torch.exp(beta * (s_i[neg] - base))).sum()
            else:
                neg_term = 0.0
            loss = loss + (pos_term + neg_term)
        return loss / B2

    @autocast()
    def forward(self, query_toks1, query_toks2, labels, mh=None):
        h1 = self.encoder(**query_toks1, return_dict=True).last_hidden_state
        h2 = self.encoder(**query_toks2, return_dict=True).last_hidden_state
        if self.agg_mode == "cls":
            e1, e2 = h1[:,0], h2[:,0]
        elif self.agg_mode == "mean_all_tok":
            e1, e2 = h1.mean(1), h2.mean(1)
        elif self.agg_mode == "mean":
            e1 = (h1 * query_toks1['attention_mask'].unsqueeze(-1)).sum(1) / query_toks1['attention_mask'].sum(-1).unsqueeze(-1)
            e2 = (h2 * query_toks2['attention_mask'].unsqueeze(-1)).sum(1) / query_toks2['attention_mask'].sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError()

        emb = torch.cat([e1, e2], dim=0)
        labels2 = torch.cat([labels, labels], dim=0)

        # use type path only if we truly have bits
        use_type_path = (mh is not None) and (mh.numel() > 0) and (mh.sum() > 0) and self.use_type_weights

        if use_type_path:
            mh = mh.to(emb.device)
            loss_ms = self._ms_loss_with_type_weights(emb, labels2, mh)
        else:
            if self.use_miner and self.miner is not None:
                hard_pairs = self.miner(emb, labels2)
                loss_ms = self.base_loss(emb, labels2, hard_pairs)
            else:
                loss_ms = self.base_loss(emb, labels2)

        loss_aux = 0.0
        use_aux = self.use_sty_aux and (self.sty_head is not None) and (mh is not None) and (mh.numel() > 0) and (mh.sum() > 0)
        if use_aux:
            mh2 = torch.cat([mh, mh], dim=0).to(emb.device)
            logits = self.sty_head(emb)
            loss_aux = F.binary_cross_entropy_with_logits(logits, mh2) * self.lambda_sty

        return loss_ms + loss_aux

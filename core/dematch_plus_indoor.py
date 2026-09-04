import torch
import torch.nn as nn
import copy


def elu_feature_map(x):
    return nn.functional.elu(x) + 1


def normalize_prior(x, eps=1e-6):
    x_min = x.amin(dim=-1, keepdim=True)
    x_max = x.amax(dim=-1, keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def estimate_point_uniqueness(points):
    # Use nearest-neighbor spacing as a lightweight proxy for single-image uniqueness.
    batch_size, _, num_pts = points.shape
    if num_pts < 2:
        return points.new_ones(batch_size, 1, num_pts)

    pts = points.transpose(1, 2)
    dist = torch.cdist(pts, pts)
    diag_mask = torch.eye(num_pts, device=points.device, dtype=torch.bool).unsqueeze(0)
    dist = dist.masked_fill(diag_mask, float("inf"))
    nearest_dist = dist.min(dim=-1).values.unsqueeze(1)
    return normalize_prior(nearest_dist)


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)

    return idx[:, :, :]

def get_graph_feature(x, k=20, idx=None):
    # x: BCN
    batch_size = x.size(0)
    num_points = x.size(2)
    # x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx_out = knn(x, k=k)
    else:
        idx_out = idx
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx_out + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

class AttentionPropagation(nn.Module):
    def __init__(
        self,
        channels,
        head,
        mode='full',
        prior_bias_scale=1.0,
        prior_bias_after_scale=False,
    ):
        nn.Module.__init__(self)
        self.head = head
        self.mode = mode
        self.head_dim = channels // head
        self.prior_bias_scale = prior_bias_scale
        self.prior_bias_after_scale = prior_bias_after_scale
        if mode=='linear':
            self.feature_map = elu_feature_map
            self.eps = 1e-6

        self.query_filter, self.key_filter, self.value_filter = nn.Conv1d(channels, channels, kernel_size=1),\
                                                              nn.Conv1d(channels, channels, kernel_size=1),\
                                                              nn.Conv1d(channels, channels, kernel_size=1)
        self.mh_filter = nn.Conv1d(channels, channels, kernel_size=1)
        self.cat_filter = nn.Sequential(
            nn.Conv1d(2*channels, 2*channels, kernel_size=1),
            nn.BatchNorm1d(2*channels), nn.ReLU(inplace=True),
            nn.Conv1d(2*channels, channels, kernel_size=1),
        )

    def forward(self, x1, x2, kv_mask=None, local=False, kv_bias=None):
        # x1(q) attend to x2(k,v)
        batch_size = x1.shape[0]
        query, key, value = self.query_filter(x1).view(batch_size, self.head, self.head_dim, -1),\
                            self.key_filter(x2).view(batch_size, self.head, self.head_dim, -1),\
                            self.value_filter(x2).view(batch_size, self.head, self.head_dim, -1)

        if self.mode == 'full':
            QK = torch.einsum('bhdn,bhdm->bhnm', query, key)
            # set masked position to -1e6
            if kv_mask is not None:
                QK.masked_fill_(~(kv_mask[:, None, None, :]), float(-1e6))
            if kv_bias is not None and not self.prior_bias_after_scale:
                QK = QK + self.prior_bias_scale * kv_bias[:, None, None, :]
            score_ = QK / self.head_dim ** 0.5
            if kv_bias is not None and self.prior_bias_after_scale:
                score_ = score_ + self.prior_bias_scale * kv_bias[:, None, None, :]
            score = torch.softmax(score_, dim=-1) # BHNM
            add_value = torch.einsum('bhnm,bhdm->bhdn', score, value).reshape(batch_size, self.head_dim * self.head, -1)
            if local:
                cluster_score = torch.mean(score, dim=1, keepdim=False) # BNM
                decluster_score = torch.mean(torch.softmax(score_, dim=-2), dim=1, keepdim=False).transpose(-1,-2) # BMN
            # assign_mat = torch.mean(torch.softmax(QK/self.head_dim**0.5,dim=-2),dim=1,keepdim=False).permute(0,2,1) # BMN
        elif self.mode == 'linear':
            # set masked position to zero
            if kv_mask is not None:
                key = key * kv_mask[:, None, None, :]
                value = value * kv_mask[:, None, None, :]
            if kv_bias is not None:
                gating = 1.0 + self.prior_bias_scale * kv_bias[:, None, None, :]
                key = key * gating
                value = value * gating
            Q = self.feature_map(query) # BHDN
            K = self.feature_map(key) # BHDM
            v_length = value.shape[-1] # BHVM
            value = value / v_length  # prevent fp16 overflow
            KV = torch.einsum("bhdm,bhvm->bhdv", K, value)
            Z = 1 / (torch.einsum("bhdn,bhd->bhn", Q, K.sum(dim=-1)) + self.eps)
            add_value = torch.einsum("bhdn,bhdv,bhn->bhvn", Q, KV, Z).reshape(batch_size, self.head_dim * self.head, -1) * v_length # B(HD)N
        else:
            raise KeyError

        add_value = self.mh_filter(add_value)
        x1_new = x1 + self.cat_filter(torch.cat([x1, add_value], dim=1))
        if local:   
            return x1_new, cluster_score, decluster_score
        else:
            return x1_new


class PositionEncoder(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.position_encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=1), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, channels, kernel_size=1)
        )
        
    def forward(self, x):
        return self.position_encoder(x)


class InitProject(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.init_project = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=1), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, channels, kernel_size=1)
        )
        
    def forward(self, x):
        return self.init_project(x)


class InlinerPredictor(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.inlier_pre = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=1), nn.InstanceNorm1d(64, eps=1e-3), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 16, kernel_size=1), nn.InstanceNorm1d(16, eps=1e-3), nn.BatchNorm1d(16), nn.ReLU(inplace=True),
            nn.Conv1d(16, 4, kernel_size=1), nn.InstanceNorm1d(4, eps=1e-3), nn.BatchNorm1d(4), nn.ReLU(inplace=True),
            nn.Conv1d(4, 1, kernel_size=1)
        )

    def forward(self, d):
        # BCN -> B1N
        return self.inlier_pre(d)


class LayerBlock(nn.Module):
    def __init__(
        self,
        channels,
        head,
        layer_names,
        cluster_top_m,
        neighbor_num,
        mode='full',
        local=False,
        decrease_flag=False,
        prior_bias_scale=1.0,
        prior_bias_after_scale=False,
    ):
        nn.Module.__init__(self)
        self.layer_names = layer_names
        self.cluster_top_m = cluster_top_m
        self.neighbor_num = neighbor_num
        self.local = local
        self.decrease_flag = decrease_flag
        if self.local:
            self.local_conv = nn.Sequential(
                nn.InstanceNorm2d(channels*2, eps=1e-3),
                nn.BatchNorm2d(channels*2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels*2, channels, kernel_size=1),
            )
            self.merge = nn.Sequential(
                nn.Conv1d(channels*2, channels, kernel_size=1), nn.InstanceNorm1d(channels, eps=1e-3), nn.BatchNorm1d(channels), nn.ReLU(inplace=True),
                nn.Conv1d(channels, channels, kernel_size=1)
            )
            
            # # Initialize the parameters of self.merge to be zero
            # for module in self.merge:
            #     if isinstance(module, nn.Conv1d):
            #         init.constant_(module.weight, 0)
            #         if module.bias is not None:
            #             init.constant_(module.bias, 0)
        
        cluster_layer = AttentionPropagation(
            channels,
            head,
            mode=mode,
            prior_bias_scale=prior_bias_scale,
            prior_bias_after_scale=prior_bias_after_scale,
        )
        self.cluster_layers = nn.ModuleList([copy.deepcopy(cluster_layer) for _ in range(len(layer_names))])
        self.inlier_pre_new = InlinerPredictor(channels)
       
    def local_context(self, d, cluster_score):
        # d: BCN, cluster_score: BPN
        B, C, N = d.shape
        self.cluster_top_m = N if self.cluster_top_m > N else self.cluster_top_m
        _, top_index = torch.topk(cluster_score, k=self.cluster_top_m, dim=-1) # BPM
        top_index = top_index.reshape(-1, self.cluster_top_m) # (BP)M
        d_cluster = torch.gather(d.unsqueeze(1).repeat(1,cluster_score.shape[1],1,1).reshape(-1,C,N), dim=-1, index=top_index.reshape(-1,1,self.cluster_top_m).repeat(1,C,1)) # (BP)CM
        d_cluster_local = get_graph_feature(d_cluster, k=self.neighbor_num) # (BP)(2*C)MK
        d_cluster_local = self.local_conv(d_cluster_local) # (BP)CMK
        d_cluster, _ = d_cluster_local.max(dim=-1, keepdim=False) # (BP)CM
        d = torch.scatter(d.unsqueeze(1).repeat(1,cluster_score.shape[1],1,1).reshape(-1,C,N), dim=-1, index=top_index.reshape(-1,1,self.cluster_top_m).repeat(1,C,1), src=d_cluster) # (BP)CN
        # _, d_top_index = torch.topk(cluster_score, k=1, dim=-2) # B1N
        # d = torch.gather(d.reshape(B,-1,C,N), dim=1, index=d_top_index.reshape(B,1,1,N).repeat(1,1,C,1)).squeeze(1)
        d, _ = d.reshape(B,-1,C,N).max(dim=1, keepdim=False) # B(LC)N -> BCN
        return d
           
    def forward(
        self,
        xs,
        d,
        feat_piece,
        logits=None,
        loss=False,
        cluster_score=None,
        decluster_score=None,
        topk_index_list=[],
        corr_prior=None,
        apply_prior_logit_bias=False,
        prior_logit_scale=1.0,
    ):
        # xs: B1N4, d: BCN, feat_piece: BCP, logits: BN
        B, C, N = d.shape
        d_old = d.clone()
        if logits is not None:
            weights = torch.relu(torch.tanh(logits)) # BN
            mask = weights>0 # BN
        else:
            mask = None
        if len(topk_index_list)>0:
            for topk_index in topk_index_list:
                feat_piece = torch.gather(feat_piece, dim=-1, index=topk_index.unsqueeze(-2).repeat(1,C,1)) # BCL
        if cluster_score is not None:
            # cluster_score: BPN
            d_local = self.local_context(d.clone().detach(), cluster_score)
        for layer, name in zip(self.cluster_layers, self.layer_names):
            if name == 'cluster':
                feat_piece, cluster_score, decluster_score = layer(feat_piece, d, mask, local=True, kv_bias=corr_prior) # BCP, BPN
            elif name == 'context':
                feat_piece = layer(feat_piece, feat_piece) # BCP
            elif name == 'decluster':
                d = layer(d, feat_piece) # BCN, BNP
            else:
                raise KeyError

        if self.decrease_flag:
            with torch.no_grad():
                assert decluster_score is not None
                # decluster_score: BNP
                _, max_index = torch.topk(decluster_score, k=1, dim=-1) # BN1
                assign_mat = torch.zeros_like(decluster_score).to(decluster_score.device) # BNP
                assign_mat = torch.scatter(assign_mat, dim=-1, index=max_index, src=torch.ones((B,N,1)).to(assign_mat.device)) # BNP
                if mask is not None:
                    assign_mat = assign_mat * mask.unsqueeze(-1)
                assign_sum = torch.sum(assign_mat, dim=-2) # BP
                assign_sum_mask = assign_sum>0 # BP
                assign_sum_mask_num = torch.sum(assign_sum_mask, dim=-1) # B
                assign_sum_mask_num_max, _ = torch.max(assign_sum_mask_num, dim=-1)
                # assign_sum_topk_value, _ = torch.topk(assign_sum, k=self.cluster_list[cluster_list_index], dim=-1) # BL
                assign_sum_topk_value, _ = torch.topk(assign_sum, k=assign_sum_mask_num_max, dim=-1) # BL
                cluster_half = torch.sum(torch.sum(assign_sum,dim=-1)==torch.sum(assign_sum_topk_value,dim=-1))==B
            if cluster_half:
                if mask is not None:
                    decluster_score_mask = decluster_score * mask.unsqueeze(-1)
                else:
                    decluster_score_mask = decluster_score
                assign_sum_soft = torch.sum(decluster_score_mask, dim=-2) # BP
                # _, topk_index_new = torch.topk(assign_sum, k=self.cluster_list[cluster_list_index], dim=-1) # BL
                # _, topk_index_new = torch.topk(assign_sum, k=assign_sum_mask_num_max, dim=-1) # BL
                _, topk_index_new = torch.topk(assign_sum_soft, k=assign_sum_mask_num_max if assign_sum_mask_num_max<decluster_score.shape[-1] else decluster_score.shape[-1], dim=-1) # BL
                # cluster_list_index += 1
            else:
                topk_index_new = None
        else:
            topk_index_new = None

        d_merge = d.clone()
        if self.local:
            d_merge = self.merge(torch.cat([d_merge, d_local], dim=-2))
        # BCN -> B1N -> BN
        logits = torch.squeeze(self.inlier_pre_new(d_merge-d_old), 1) # BN
        if apply_prior_logit_bias and corr_prior is not None:
            logits = logits + prior_logit_scale * (corr_prior * 2.0 - 1.0)
        if loss:
            e_hat = weighted_8points(xs, logits)
        else:
            e_hat = None
        return d, logits, e_hat, cluster_score, decluster_score, topk_index_new


class DeMatchPlus(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.layer_num = config.layer_num
        self.use_prior_input = getattr(config, 'use_prior_input', False)
        self.use_prior_soft_bias = getattr(config, 'use_prior_soft_bias', False)
        self.use_prior_logit_bias = getattr(config, 'use_prior_logit_bias', False)
        self.use_signed_prior_mask_attention = getattr(config, 'use_signed_prior_mask_attention', False)
        self.signed_prior_mask_scale = getattr(config, 'signed_prior_mask_scale', 1.0)
        self.prior_use_uniqueness = getattr(config, 'prior_use_uniqueness', True)
        self.prior_use_side_matchability = getattr(config, 'prior_use_side_matchability', True)
        self.prior_bias_scale = getattr(config, 'prior_bias_scale', 1.0)
        self.prior_logit_scale = getattr(config, 'prior_logit_scale', 1.0)
        self.prior_logit_bias_last_only = getattr(config, 'prior_logit_bias_last_only', False)
        self.config_use_ratio = getattr(config, 'use_ratio', 0)
        self.config_use_mutual = getattr(config, 'use_mutual', 0)
        if self.use_signed_prior_mask_attention:
            self.signed_prior_mask_alpha = nn.Parameter(torch.ones(1))
        else:
            self.register_parameter('signed_prior_mask_alpha', None)

        self.piece_tokens = nn.Parameter(torch.randn(config.net_channels, config.piece_num)) # CP
        self.register_parameter('piece_tokens', self.piece_tokens)
        self.pos_embed = PositionEncoder(config.net_channels)
        self.init_project = InitProject(config.net_channels)
        prior_in_dim = self._get_prior_in_dim(config)
        if self.use_prior_input and prior_in_dim > 0:
            hidden_dim = max(config.net_channels // 2, 16)
            self.prior_proj = nn.Sequential(
                nn.Conv1d(prior_in_dim, hidden_dim, kernel_size=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_dim, config.net_channels, kernel_size=1),
            )
        else:
            self.prior_proj = None
        self.layer_blocks = nn.Sequential(
            *[
                LayerBlock(
                    config.net_channels,
                    config.head,
                    config.layer_names,
                    config.cluster_top_num,
                    config.neighbor_num,
                    mode=config.attention_mode,
                    local=layer_index > 0,
                    decrease_flag=layer_index in config.decrease_layer,
                    prior_bias_scale=self.prior_bias_scale,
                    prior_bias_after_scale=self.use_signed_prior_mask_attention,
                )
                for layer_index in range(self.layer_num)
            ]
        )
        self.latest_prior_stats = {}
        
        nn.init.uniform_(self.piece_tokens)

    def _set_latest_prior_stats(self, stats):
        self.latest_prior_stats = {}
        for key, value in stats.items():
            if isinstance(value, torch.Tensor):
                self.latest_prior_stats[key] = float(value.detach().mean().cpu())
            else:
                self.latest_prior_stats[key] = float(value)

    def _get_prior_in_dim(self, config):
        prior_in_dim = 0
        if self.prior_use_uniqueness:
            prior_in_dim += 3
        if self.prior_use_side_matchability:
            if config.use_ratio == 2:
                prior_in_dim += 1
            if config.use_mutual == 2:
                prior_in_dim += 1
            prior_in_dim += 1
        if self.prior_use_uniqueness and self.prior_use_side_matchability:
            prior_in_dim += 1
        return prior_in_dim

    def _use_prior_logit_bias_for_layer(self, layer_index):
        if not self.use_prior_logit_bias:
            return False
        if self.prior_logit_bias_last_only:
            return layer_index == self.layer_num - 1
        return True

    def _build_attention_prior(self, corr_prior):
        if corr_prior is None:
            return None
        if not self.use_signed_prior_mask_attention:
            self.latest_prior_stats["prior/signed_mask_enabled"] = 0.0
            return corr_prior
        signed_mask = (corr_prior * 2.0 - 1.0) * self.signed_prior_mask_scale
        signed_mask = signed_mask * self.signed_prior_mask_alpha
        self.latest_prior_stats["prior/signed_mask_enabled"] = 1.0
        self.latest_prior_stats["prior/signed_mask_mean"] = float(signed_mask.mean().detach().cpu())
        self.latest_prior_stats["prior/signed_mask_alpha"] = float(self.signed_prior_mask_alpha.detach().cpu())
        return signed_mask

    def _build_prior(self, data, x1, x2):
        components = []
        joint_uniqueness = None
        stats = {}

        if self.prior_use_uniqueness:
            uniq0 = estimate_point_uniqueness(x1)
            uniq1 = estimate_point_uniqueness(x2)
            joint_uniqueness = normalize_prior(uniq0 * uniq1)
            components.extend([uniq0, uniq1, joint_uniqueness])
            stats["prior/uniq0_mean"] = uniq0.mean()
            stats["prior/uniq1_mean"] = uniq1.mean()
            stats["prior/joint_uniqueness_mean"] = joint_uniqueness.mean()

        matchability_terms = []
        if self.prior_use_side_matchability:
            sides = data.get('sides')
            side_offset = 0
            if isinstance(sides, list) and len(sides) == 0:
                sides = None
            if torch.is_tensor(sides):
                sides = sides.transpose(1, 2)
            else:
                sides = None

            if self.config_use_ratio == 2 and sides is not None and sides.shape[1] > side_offset:
                ratio_prior = 1.0 - sides[:, side_offset:side_offset + 1, :].clamp(0, 1)
                ratio_prior = normalize_prior(ratio_prior)
                components.append(ratio_prior)
                matchability_terms.append(ratio_prior)
                stats["prior/ratio_mean"] = ratio_prior.mean()
                side_offset += 1
            if self.config_use_mutual == 2 and sides is not None and sides.shape[1] > side_offset:
                mutual_prior = sides[:, side_offset:side_offset + 1, :].clamp(0, 1)
                mutual_prior = normalize_prior(mutual_prior)
                components.append(mutual_prior)
                matchability_terms.append(mutual_prior)
                stats["prior/mutual_mean"] = mutual_prior.mean()

            if matchability_terms:
                matchability = normalize_prior(torch.stack(matchability_terms, dim=0).mean(dim=0))
            else:
                matchability = x1.new_ones(x1.shape[0], 1, x1.shape[-1])
            components.append(matchability)
            stats["prior/matchability_mean"] = matchability.mean()

            if joint_uniqueness is not None:
                fused_prior = normalize_prior(joint_uniqueness * matchability)
                components.append(fused_prior)
                stats["prior/fused_mean"] = fused_prior.mean()

        if not components:
            self._set_latest_prior_stats({
                "prior/enabled": 0.0,
                "prior/input_enabled": 1.0 if self.use_prior_input else 0.0,
                "prior/soft_bias_enabled": 1.0 if self.use_prior_soft_bias else 0.0,
                "prior/logit_bias_enabled": 1.0 if self.use_prior_logit_bias else 0.0,
            })
            return None
        prior = torch.cat(components, dim=1)
        stats["prior/enabled"] = 1.0
        stats["prior/input_enabled"] = 1.0 if self.use_prior_input else 0.0
        stats["prior/soft_bias_enabled"] = 1.0 if self.use_prior_soft_bias else 0.0
        stats["prior/logit_bias_enabled"] = 1.0 if self.use_prior_logit_bias else 0.0
        stats["prior/feature_mean"] = prior.mean()
        self._set_latest_prior_stats(stats)
        return prior

    def forward(self, data, training=False):
        assert data['xs'].dim() == 4 and data['xs'].shape[1] == 1
        batch_size, num_pts = data['xs'].shape[0], data['xs'].shape[2]
        # B1NC -> BCN
        input = data['xs'].transpose(1,3).squeeze(3) # B4N
        x1, x2 = input[:,:2,:], input[:,2:,:]
        motion = x2 - x1 # B2N

        pos = x1 # B2N
        pos_embed = self.pos_embed(pos) # BCN

        d = self.init_project(motion) + pos_embed # BCN
        prior = self._build_prior(data, x1, x2)
        corr_prior = None
        if prior is not None:
            corr_prior = normalize_prior(prior.mean(dim=1))
            self.latest_prior_stats["prior/corr_prior_mean"] = float(corr_prior.mean().detach().cpu())
            if self.prior_proj is not None:
                d = d + self.prior_proj(prior)
        else:
            self.latest_prior_stats["prior/corr_prior_mean"] = 0.0
        attention_prior = self._build_attention_prior(corr_prior)
        feat_piece, _ = torch.linalg.qr(self.piece_tokens, mode='reduced') # QR decomposition
        feat_piece = feat_piece.unsqueeze(0).repeat(batch_size, 1, 1) # CP->BCP

        res_logits, res_e_hat = [], []
        logits = None
        cluster_score = None
        decluster_score = None
        topk_index_list = []

        for i in range(self.layer_num):
            if i<self.layer_num-1:
                d, logits, e_hat, cluster_score, decluster_score, topk_index = self.layer_blocks[i](
                    data['xs'],
                    d,
                    feat_piece,
                    logits=logits,
                    loss=training,
                    cluster_score=cluster_score,
                    decluster_score=decluster_score,
                    topk_index_list=topk_index_list,
                    corr_prior=attention_prior if self.use_prior_soft_bias else None,
                    apply_prior_logit_bias=self._use_prior_logit_bias_for_layer(i),
                    prior_logit_scale=self.prior_logit_scale,
                ) # BCN
                if topk_index is not None:
                    topk_index_list.append(topk_index)
                res_logits.append(logits), res_e_hat.append(e_hat)
            else:
                d, logits, e_hat, _, _, _ = self.layer_blocks[i](
                    data['xs'],
                    d,
                    feat_piece,
                    logits=logits,
                    loss=True,
                    cluster_score=cluster_score,
                    decluster_score=decluster_score,
                    topk_index_list=topk_index_list,
                    corr_prior=attention_prior if self.use_prior_soft_bias else None,
                    apply_prior_logit_bias=self._use_prior_logit_bias_for_layer(i),
                    prior_logit_scale=self.prior_logit_scale,
                ) # BCN
                res_logits.append(logits), res_e_hat.append(e_hat)
        if logits is not None:
            self.latest_prior_stats["prior/final_logit_mean"] = float(logits.mean().detach().cpu())
        return res_logits, res_e_hat


def batch_symeig(X):
    # it is much faster to run symeig on CPU
    device = X.device
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b,d,d)
    for batch_idx in range(X.shape[0]):
        # e,v = torch.symeig(X[batch_idx,:,:].squeeze(), True)
        e,v = torch.linalg.eigh(X[batch_idx,:,:].squeeze(), UPLO='U')
        bv[batch_idx,:,:] = v
    return bv.to(device)


def weighted_8points(x_in, logits):
    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape
    # Turn into weights for each sample
    weights = torch.relu(torch.tanh(logits))
    x_in = x_in.squeeze(1)
    
    # Make input data (num_img_pair x num_corr x 4)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1)

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1)
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1), wX)
    

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat

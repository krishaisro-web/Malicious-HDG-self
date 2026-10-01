import torch
import torch.nn as nn
from src.hdg.models.legacy.encoder import FeatureProjection, make_hetero_layer, ALL_RELATIONS
from src.hdg.models.legacy.classifier import Classifier

class FullModel(nn.Module):
    def __init__(self, hidden_dim=64, dropout=0.3, relations=None):
        super().__init__()
        self.relations = relations if relations is not None else ALL_RELATIONS
        self.proj = FeatureProjection(hidden_dim)
        self.layer1 = make_hetero_layer(hidden_dim, self.relations)
        self.layer2 = make_hetero_layer(hidden_dim, self.relations)
        self.dropout = nn.Dropout(dropout)
        self.classifier = Classifier(hidden_dim)

    def encode_snapshot(self, data):
        x_dict = self.proj(data.x_dict)
        x_dict = self.layer1(x_dict, data.edge_index_dict)
        x_dict = {k: torch.relu(v) for k, v in x_dict.items()}
        x_dict = self.layer2(x_dict, data.edge_index_dict)
        x_dict = {k: self.dropout(torch.relu(v)) for k, v in x_dict.items()}
        return x_dict["domain"]

    def forward(self, snapshot_list, domain_snapshot_id):
        per_snap_embeds = [self.encode_snapshot(s) for s in snapshot_list]
        stacked = torch.stack(per_snap_embeds, dim=1)   # [num_domains, num_snapshots, hidden]
        hidden_dim = stacked.size(-1)
        idx = domain_snapshot_id.view(-1, 1, 1).expand(-1, 1, hidden_dim)
        gathered = torch.gather(stacked, 1, idx).squeeze(1)   # [num_domains, hidden] — each domain's OWN snapshot
        return self.classifier(gathered)

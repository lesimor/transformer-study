from copy import deepcopy
from torch import nn


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, h, qkv_fc_layer, fc_layer):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.query_fc_layer = deepcopy(qkv_fc_layer)
        self.key_fc_layer = deepcopy(qkv_fc_layer)
        self.value_fc_layer = deepcopy(qkv_fc_layer)
        self.fc_layer = fc_layer

    def forward(self, query, key, value, mask=None):
        n_batch = query.shape[0]

        def transform(x, fc_layer):
            out = fc_layer(x)  # shape: (n_batch, seq_len, d_model)
            out = out.view(
                n_batch, -1, self.h, self.d_model // self.h
            )  # shape: (n_batch, seq_len, h, d_k)
            out = out.transpose(1, 2)  # shape: (n_batch, h, seq_len, d_k)
        
        query = transform(query, self.query_fc_layer)
        key = transform(key, self.key_fc_layer)
        value = transform(value, self.value_fc_layer)

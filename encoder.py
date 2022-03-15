from torch import nn
from copy import deepcopy


class Encoder(nn.Module):
    def __init__(self, encoder_layer, n_layer):
        """인코더 클래스

        Args:
            encoder_layer (Encoder): 인코더 레이어 인스턴스
            n_layer (int): 인코더 레이어 갯수
        """
        super(Encoder, self).__init__()
        self.layers = []

        for i in range(n_layer):
            # 왜 딥카피를 하는 것일까?
            # deepcopy를 호출하는 이유는 실제로는 서로 다른 weight를 갖고 별개로 운용되게 하기 위함
            self.layers.append(deepcopy(encoder_layer))

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, multi_head_attention_layer, position_wise_feed_forward_layer):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention_layer = multi_head_attention_layer
        self.position_wise_feed_forward_layer = position_wise_feed_forward_layer

    def forward(self, x):
        out = self.multi_head_attention_layer(x)
        out = self.position_wise_feed_forward_layer(out)
        return out

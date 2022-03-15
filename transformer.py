from torch import nn


class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, z):
        c = self.encoder(x)
        y = self.decoder(c, z)
        return y

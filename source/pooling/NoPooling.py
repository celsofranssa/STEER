from pytorch_lightning import LightningModule

class NoPooling(LightningModule):
    """
    Performs no pooling on the transformer output.
    """

    def __init__(self):
        super(NoPooling, self).__init__()

    def forward(self, encoder_outputs, attention_mask):
        return encoder_outputs.pooler_output

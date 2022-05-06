from torch import nn
from models.biaffine import Biaffine


class BiaffineEdgeScorer(nn.Module):
    def __init__(self, rnn_size, mlp_size):
        super().__init__()

        mlp_activation = nn.ReLU()

        # The two MLPs that we apply to the RNN output before the biaffine scorer.
        self.arc_head_mlp = nn.Sequential(nn.Linear(rnn_size, mlp_size), mlp_activation)
        self.arc_dep_mlp = nn.Sequential(nn.Linear(rnn_size, mlp_size), mlp_activation)

        self.arc_attn = Biaffine(
            n_in=mlp_size,
            scale=0,
            bias_x=True,
            bias_y=False
        )

    def forward(self, sentence_repr):
        H_arc_head = self.arc_head_mlp(sentence_repr)
        H_arc_dep = self.arc_dep_mlp(sentence_repr)

        return self.arc_attn(H_arc_dep, H_arc_head)

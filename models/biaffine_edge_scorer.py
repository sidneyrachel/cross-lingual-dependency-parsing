from torch import nn
from models.biaffine import Biaffine


class BiaffineEdgeScorer(nn.Module):
    def __init__(
            self,
            rnn_size,
            arc_mlp_size,
            rel_mlp_size,
            rel_size
    ):
        super().__init__()

        mlp_activation = nn.ReLU()

        # The two MLPs that we apply to the RNN output before the biaffine scorer.
        self.arc_head_mlp = nn.Sequential(nn.Linear(rnn_size, arc_mlp_size), mlp_activation)
        self.arc_dep_mlp = nn.Sequential(nn.Linear(rnn_size, arc_mlp_size), mlp_activation)
        self.rel_head_mlp = nn.Sequential(nn.Linear(rnn_size, rel_mlp_size), mlp_activation)
        self.rel_dep_mlp = nn.Sequential(nn.Linear(rnn_size, rel_mlp_size), mlp_activation)

        self.arc_attn = Biaffine(
            n_in=arc_mlp_size,
            scale=0,
            bias_x=True,
            bias_y=False
        )
        self.rel_attn = Biaffine(
            n_in=rel_mlp_size,
            n_out=rel_size,
            bias_x=True,
            bias_y=True
        )

    def forward(self, sentence_repr):
        H_arc_head = self.arc_head_mlp(sentence_repr)
        H_arc_dep = self.arc_dep_mlp(sentence_repr)
        H_rel_head = self.rel_head_mlp(sentence_repr)
        H_rel_dep = self.rel_dep_mlp(sentence_repr)

        s_arc = self.arc_attn(H_arc_dep, H_arc_head)
        s_rel = self.rel_attn(H_rel_dep, H_rel_head).permute(0, 2, 3, 1)

        return s_arc, s_rel

import torch
import torch.nn as nn
import geoopt.geoopt.manifolds.poincare.math as pmath
# import geoopt
from geoopt.geoopt.manifolds.euclidean import Euclidean
from geoopt.geoopt.manifolds.poincare import PoincareBall
# import hyrnn
import functools
from hyrnn.hyrnn.lookup_embedding import LookupEmbedding
from hyrnn.hyrnn.nets import MobiusGRU
# Question: Q

class RNNBase(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        project_dim,
        cell_type="eucl_rnn",
        embedding_type="eucl",
        decision_type="eucl",
        use_distance_as_feature=True,
        device=None,
        num_layers=1,
        num_classes=1,
        c=1.0,
    ):
        super(RNNBase, self).__init__()
        (cell_type, embedding_type, decision_type) = map(
            str.lower, [cell_type, embedding_type, decision_type]
        )
        if embedding_type == "eucl":
            self.embedding = LookupEmbedding(
                vocab_size, embedding_dim, manifold=Euclidean()
            )
            with torch.no_grad():
                self.embedding.weight.normal_()
        elif embedding_type == "hyp":
            self.embedding = LookupEmbedding(
                vocab_size,
                embedding_dim,
                manifold=PoincareBall(c=c),
            )
            with torch.no_grad():
                self.embedding.weight.set_(
                    pmath.expmap0(self.embedding.weight.normal_() / 10, c=c) # Q
                )
        else:
            raise NotImplementedError(
                "Unsuported embedding type: {0}".format(embedding_type)
            )
        self.embedding_type = embedding_type
        if decision_type == "eucl": 
            self.projector = nn.Linear(hidden_dim * 2, project_dim) # Q
            self.logits = nn.Linear(project_dim, num_classes)
        elif decision_type == "hyp":
            self.projector_source = hyrnn.MobiusLinear( # Q
                hidden_dim, project_dim, c=c
            )
            self.projector_target = hyrnn.MobiusLinear( # Q
                hidden_dim, project_dim, c=c
            )
            self.logits = hyrnn.MobiusDist2Hyperplane(project_dim, num_classes) # Q
        else:
            raise NotImplementedError(
                "Unsuported decision type: {0}".format(decision_type)
            )
        self.ball = PoincareBall(c)
        if use_distance_as_feature:
            if decision_type == "eucl":
                self.dist_bias = nn.Parameter(torch.zeros(project_dim))
            else:
                self.dist_bias = geoopt.ManifoldParameter(
                    torch.zeros(project_dim), manifold=self.ball
                )
        else:
            self.register_buffer("dist_bias", None)
        self.decision_type = decision_type
        self.use_distance_as_feature = use_distance_as_feature
        self.device = device  # declaring device here due to fact we are using catalyst
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.c = c

        if cell_type == "eucl_rnn":
            self.cell = nn.RNN
        elif cell_type == "eucl_gru":
            self.cell = nn.GRU
        elif cell_type == "hyp_gru":
            self.cell = functools.partial(MobiusGRU, c=c)
        else:
            raise NotImplementedError("Unsuported cell type: {0}".format(cell_type))
        self.cell_type = cell_type

        self.cell_source = self.cell(embedding_dim, self.hidden_dim, self.num_layers)
        self.cell_target = self.cell(embedding_dim, self.hidden_dim, self.num_layers)

    def forward(self, input):

        source_input = input[0][0]
        # print(source_input)
        target_input = input[0][1]
        alignment = input[1]
        batch_size = alignment.shape[0]

        source_input_data = self.embedding(source_input.data)
        target_input_data = self.embedding(target_input.data)

        zero_hidden = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_dim,
            device=self.device or source_input.device,
            dtype=source_input_data.dtype
        )
        # print(self.cell_type)
        if self.embedding_type == "eucl" and "hyp" in self.cell_type: # This is for the example
            # print(source_input_data.shape)
            source_input_data = pmath.expmap0(source_input_data, c=self.c)
            # print(source_input_data.shape)
            target_input_data = pmath.expmap0(target_input_data, c=self.c)
        elif self.embedding_type == "hyp" and "eucl" in self.cell_type:
            source_input_data = pmath.logmap0(source_input_data, c=self.c)
            target_input_data = pmath.logmap0(target_input_data, c=self.c)
        # ht: (num_layers * num_directions, batch, hidden_size)

        # print(source_input.batch_sizes.shape)
        source_input = torch.nn.utils.rnn.PackedSequence(
            source_input_data, source_input.batch_sizes
        )
        target_input = torch.nn.utils.rnn.PackedSequence(
            target_input_data, target_input.batch_sizes
        )

        _, source_hidden = self.cell_source(source_input, zero_hidden)
        _, target_hidden = self.cell_target(target_input, zero_hidden)

        # take hiddens from the last layer
        source_hidden = source_hidden[-1]
        # print(target_hidden)
        target_hidden = target_hidden[-1][alignment]
        # print(alignment)
        # print(target_hidden)

        if self.decision_type == "hyp":
            if "eucl" in self.cell_type:
                source_hidden = pmath.expmap0(source_hidden, c=self.c)
                target_hidden = pmath.expmap0(target_hidden, c=self.c)
            source_projected = self.projector_source(source_hidden)
            target_projected = self.projector_target(target_hidden)
            projected = pmath.mobius_add(
                source_projected, target_projected, c=self.ball.c
            )
            if self.use_distance_as_feature:
                dist = (
                    pmath.dist(source_hidden, target_hidden, dim=-1, keepdim=True, c=self.ball.c) ** 2
                )
                bias = pmath.mobius_scalar_mul(dist, self.dist_bias, c=self.ball.c)
                projected = pmath.mobius_add(projected, bias, c=self.ball.c)
        else:
            if "hyp" in self.cell_type:
                source_hidden = pmath.logmap0(source_hidden, c=self.c)
                target_hidden = pmath.logmap0(target_hidden, c=self.c)
            projected = self.projector(
                torch.cat((source_hidden, target_hidden), dim=-1)
            )
            if self.use_distance_as_feature:
                dist = torch.sum(
                    (source_hidden - target_hidden).pow(2), dim=-1, keepdim=True
                )
                bias = self.dist_bias * dist
                projected = projected + bias

        logits = self.logits(projected)
        # CrossEntropy accepts logits
        return logits

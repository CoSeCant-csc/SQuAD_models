import torch
from torch.nn.parameter import Parameter
from overrides import overrides

from drqa.module.similarity_functions.similarity_function import SimilarityFunction


class SymmetricBilinearSimilarity(SimilarityFunction):
    """
    This similarity function performs a bilinear transformation of the two input vectors.  This
    function have a matrix of weights ``W`` and a diagonal matrix of weights "D", and the similarity between two vectors
    ``x`` and ``y`` is computed as ``x^T W^T D W y".

    Parameters
    ----------
    tensor_dim : "int"
        The dimension of "x" described above.  This is "x.size()[-1]" - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build the weight matrix correctly.
    attention_dim : "int"
        The dimension of "D" (i.e. dimension k of "D \in {k \times k}") and the first dimension of "W"
        (i.e. dimension k of "W \in {k \times tensor_dim}")
    activation : ``Activation``, optional (default=linear (i.e. no activation))
        An activation function applied after the ``x^T W^T D W y`` calculation.  Default is no
        activation.
    """
    def __init__(self,
                 tensor_dim: int,
                 attention_dim: int,
                 activation=lambda x: x) -> None:
        super(SymmetricBilinearSimilarity, self).__init__()
        self._weight_matrix = Parameter(torch.Tensor(attention_dim, tensor_dim))
        self._diagnoal_matrix = Parameter(torch.Tensor(attention_dim))
        self._activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform(self._weight_matrix)

    @overrides
    def forward(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        intermediate = torch.matmul(tensor_1, self._weight_matrix.transpose())
        intermediate = torch.matmul(intermediate, torch.diag(self._diagnoal_matrix))
        intermediate = torch.matmul(intermediate, self._weight_matrix)
        result = (intermediate * tensor_2).sum(dim=-1)
        return self._activation(result)

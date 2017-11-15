"""
A ``SimilarityFunction`` takes a pair of tensors with the same shape, and computes a similarity
function on the vectors in the last dimension.
"""
from drqa.module.similarity_functions.bilinear import BilinearSimilarity
from drqa.module.similarity_functions.cosine import CosineSimilarity
from drqa.module.similarity_functions.dot_product import DotProductSimilarity
from drqa.module.similarity_functions.linear import LinearSimilarity
from drqa.module.similarity_functions.multiheaded import MultiHeadedSimilarity
from drqa.module.similarity_functions.similarity_function import SimilarityFunction

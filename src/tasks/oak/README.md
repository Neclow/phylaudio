# Oak - Orthogonal Additive Gaussian Processes

## Source

This code is copied from
<https://github.com/harrisonzhu508/orthogonal-additive-gaussian-processes>

- Original repository:
  <https://github.com/amzn/orthogonal-additive-gaussian-processes>
- Original paper: <https://arxiv.org/abs/2206.09861>
- Fork:
  <https://github.com/harrisonzhu508/orthogonal-additive-gaussian-processes>
- Branch: my-oak-improvements
- Date copied: December 15, 2024

## License

This code is licensed under Apache License 2.0. See LICENSE file in this
directory.

## Modifications

- Multi-dimensional kernel support: extended to handle continuous 2D orthogonal
  RBF kernels and multi-dimensional Sobol indices
- Normalizing flow functionality: Added copula transformations and Sinh-Arcsinh
  bijectors for handling non-Gaussian input distributions
- Concurvity penalty for correlated inputs
- Shapley value computation for feature importance
- Time-specific model support for temporal data
- Shared variance parameterization across interaction orders

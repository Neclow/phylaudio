# Resources

Reference language code tables and papers used in this project.

- `iso-639-3.tab` — ISO 639-3 language code registry (ID, scope, type, reference name)

## Reference papers

`plots/supplementary_v2.ipynb` parses training-hours tables directly from the
following papers. The PDFs are not committed to the repository (copyright); run
`pixi run download_papers` to fetch them into this directory from their original
sources at the exact versions the parsing code expects:

- `2111.09296.pdf` — Babu et al. 2021, XLS-R: self-supervised cross-lingual speech representation learning at scale ([arXiv:2111.09296v3](https://arxiv.org/abs/2111.09296v3))
- `mls.pdf` — Pratap et al. 2020, MLS: a large-scale multilingual dataset for speech research ([arXiv:2012.03411v2](https://arxiv.org/abs/2012.03411v2))
- `vl107.pdf` — Valk & Alumäe 2020, VoxLingua107: a dataset for spoken language recognition ([arXiv:2011.12998v1](https://arxiv.org/abs/2011.12998v1))
- `voxpopuli.pdf` — Wang et al. 2021, VoxPopuli: a large-scale multilingual speech corpus ([ACL 2021.acl-long.80](https://aclanthology.org/2021.acl-long.80/))

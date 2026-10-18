**Data availability statement**

Classical profile matching is the bar for learned powder-diffraction retrieval: a real organic benchmark against 309,080 CSD entries (F. Cai)

The code, the benchmark definition and all results are available at https://github.com/fronkt/xrd-clip and archived at Zenodo
(https://doi.org/10.5281/zenodo.23446290). The deposit contains: the opXRD pattern identifiers with their CSD refcodes, same-form sets and
packing-similarity results; the corpus refcode list and the training splits; the per-pattern ranks of
every score and correction on the measured patterns, including the learned model's and the variable-cell
re-rank's (the simulated sweep of Section 4.1 and the effect ladder of Section 4.5 are deposited as summary
results); the
critic2 inputs derived from the opXRD patterns (cut patterns, background and peak fits, command files) and
the measurement conditions of the single-crystal patterns; the scripts that build the library from a CSD
installation, compute the scores (CPU and GPU) and regenerate every table and figure; and
results/paper_numbers.json, from which every number in this article is checked by scripts/verify_claims.py.
The open replica of Section 2.6 is included in full: the COD identifiers, the simulated COD library, its
answer sets and the ranks of every score, from which the classical results of Section 4.7 and the COD
columns of Table 2 can be reproduced without a CSD licence. The CSD library was built from the May 2026
release (1,451,367 entries; CSD Python API 3.7.1) and the measurement conditions were read from a later
update (1,469,017 entries; API 3.8.0); the COD was downloaded on 12–13 October 2026; the opXRD patterns are
from the Zenodo record 10.5281/zenodo.15298026. The measured patterns are from opXRD under CC BY
4.0, and the COD structures are CC0. CSD structures and the simulated patterns derived from them cannot be
redistributed; users with a CSD licence regenerate them from the refcode lists. The model checkpoints, the
original and the retrained one, were trained on CSD data and are not distributed.
The external set of Section 4.7 uses the measured profiles in the supplementary CIFs of the cited Powder
Diffraction reports, which can be downloaded from Cambridge Core, and, for donepezil hydrochloride form III,
the CIF deposited with the ICDD; these profiles are not redistributed, and the deposit contains their
refcodes, the extraction script and the per-pattern ranks. The reports listed in ESI Table S24 are
cited in the article's reference list.

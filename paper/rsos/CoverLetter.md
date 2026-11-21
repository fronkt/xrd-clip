Dear Editor,

Please consider the enclosed manuscript, "Retrieving organic crystal structures from measured powder patterns: a benchmark of classical profile matching and a symmetry-aware contrastive model", for publication as a
Research Article in Royal Society Open Science (subject area: Chemistry; also relevant to Computer Science and
Artificial Intelligence).

Machine-learning models that embed powder diffraction patterns and crystal structures in a shared space have been
proposed for structure retrieval, but they are mostly evaluated on simulated patterns. The manuscript provides a
benchmark built from 188 measured powder patterns of organic compounds, whose correct answers in the Cambridge
Structural Database are fixed by packing similarity and searched among 309,080 simulated entries. On it, classical
profile scores that tolerate peak shifts retrieve independent single-crystal references, while a contrastive model
with a space-group-aware structure encoder, trained on simulations that include every measurement effect we could
model, does not. The model retrieves simulated versions of the same patterns, so its failure lies in differences
between measured and simulated patterns that the simulation does not reproduce. We think the journal's focus on sound
methods and openly reported results, including negative ones, fits a study of this kind.

An earlier version was submitted to RSC Advances (manuscript RA-ART-10-2026-010272) and declined on 10 November 2026
after peer review. The present manuscript is substantially revised. We fixed three defects in the model and its
training simulation, retrained it with a structure encoder that sees the space group and with two seeds, made the
ambient single-crystal set the headline, added texture along each structure's own growth axis to the simulated tests,
and added checks of the labels, of the intensity convention and of the anchor procedure. We enclose a point-by-point response to the previous reviews and a table of every change from the earlier version.

All code, the benchmark definition, an open replica built from the Crystallography Open Database and every
per-pattern rank are available at https://github.com/fronkt/xrd-clip and archived at Zenodo (https://doi.org/10.5281/zenodo.24407260). Structures from the
Cambridge Structural Database cannot be redistributed; the deposit contains the refcode lists from which licence holders
regenerate them. The manuscript has not been published and is not under consideration elsewhere. The author declares no
competing interests.

Yours sincerely,

Frank Cai
Purdue University, West Lafayette, Indiana, USA

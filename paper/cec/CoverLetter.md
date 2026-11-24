Dear Editor,

Please consider the enclosed manuscript, "Retrieving organic crystal structures from measured powder patterns: a benchmark of classical profile matching and a symmetry-aware contrastive model", for publication as a Full paper in CrystEngComm.

Identifying which solid form a powder sample contains is a routine step in crystal engineering and in pharmaceutical solid-form work. It is usually done by comparing the measured pattern with reference patterns, measured or simulated from databases such as the Cambridge Structural Database. Machine-learning models that embed powder patterns and crystal structures in a shared space have been proposed for this search, but they are evaluated mostly on simulated patterns and mostly on inorganic or framework materials. The manuscript tests the idea on measured patterns of organic molecular crystals, with each correct answer fixed by packing similarity so that polymorphs are kept apart.

The results bear directly on how forms are identified by powder diffraction. Classical profile scores that tolerate a zero offset or a lattice strain rank the independent single-crystal reference first for more than half of the compounds, while the learned model ranks it first for none, although its training simulation included every measurement effect we could model. Most of the remaining failures of the classical scores have a crystallographic cause: the reference was determined at another temperature or pressure, and the cells differ by more than about 1 % in at least one reduced-cell length, a difference that an isotropic shift or strain cannot absorb.

All code, the benchmark definition, an open replica built from the Crystallography Open Database and every per-pattern rank are available at https://github.com/fronkt/xrd-clip and archived at Zenodo (https://doi.org/10.5281/zenodo.24407260). Structures from the Cambridge Structural Database cannot be redistributed; the deposit contains the refcode lists from which licence holders regenerate them. The manuscript has not been published and is not under consideration elsewhere.

Yours sincerely,

Frank Cai
Purdue University, West Lafayette, Indiana, USA

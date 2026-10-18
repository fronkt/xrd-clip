<!-- [DRAFTED — REVIEW] Frank: every claim below is taken from the manuscript; no significance is added that the
paper does not state. Remove this comment once you have read it. Do not name reviewers here (RSC: ScholarOne only). -->

Frank Cai\
Purdue University, West Lafayette, Indiana, USA\
cai485@purdue.edu · ORCID 0009-0003-0041-1459

Dear Editor,

I submit the manuscript "Classical profile matching is the bar for learned powder-diffraction retrieval:
a real organic benchmark against 309,080 CSD entries" for consideration as a Paper in *RSC Advances*.

Identifying which known crystal form a measured powder pattern belongs to is a routine task in
pharmaceutical and organic solid-state chemistry, where polymorphs, solvates and salts of one compound must be
told apart. Machine-learned models that embed powder patterns and crystal structures in a shared space have
been proposed for this search, but they are mostly tested on simulated patterns against baselines that cannot
absorb a peak shift. This manuscript asks how such a model compares with classical profile matching on measured
patterns of real organic compounds.

The benchmark uses 188 measured patterns of 123 organic compounds from the opXRD database, with correct answers
fixed by CSD packing similarity rather than by cell or formula matches, and ranks every pattern against 309,080
simulated entries of the Cambridge Structural Database. The main findings are:

- When the answer must be an independent single-crystal determination, classical scores place a correct
  entry first for 0.30–0.50 of the compounds and the learned model for none; most failures involve references
  determined at another temperature or pressure.
- The model retrieves clean simulated patterns of the same compounds, and a retrain with five modelled
  measurement effects does not close the gap on measured patterns, so the failure lies in differences between
  measured and simulated patterns that current simulation does not reproduce.
- The conclusions hold with a more realistic library, a hubness correction, an open replica built from the
  Crystallography Open Database, and an external set of 19 synchrotron and laboratory patterns deposited with
  published pharmaceutical structure reports.

I believe the work suits *RSC Advances* because it gives chemists who identify crystal forms from powder data a
measured-data reference point for learned methods, and because the benchmark (code, refcode lists, per-pattern
ranks and the open replica) is released so that other methods can be compared on the same patterns. All numbers
in the manuscript are checked automatically against the deposited results.

The manuscript has not been published and is not under consideration elsewhere. I am the sole author, there
are no conflicts to declare, and the use of AI tools is stated in the Acknowledgements.

Sincerely,

Frank Cai

---
journal: "RSC Advances"
code: "RSCA"

page:
  size: a4
  width_cm: null
  height_cm: null
  margins_cm:
    top: 2.5
    bottom: 2.5
    left: 2.5
    right: 2.5

body:
  font: "Times New Roman"
  size_pt: 12
  line_spacing: 1.5
  space_before_pt: 0
  space_after_pt: 6
  alignment: left
  first_line_indent_cm: null
  preserve_fonts: []
  strip_highlights: true
  force_black: false

headings:
  font: null
  color: "000000"
  bold: true
  sizes_pt:
    1: 14
    2: 12
    3: 12

captions:
  size_pt: null

tables:
  line_spacing: null
  placement: inline

line_numbers: continuous
page_numbers: bottom_center
header_text: null

limits:
  abstract_words: 250
  keywords_min: null
  keywords_max: null
  pages_max: null
  words_max: null
  figures_max: null

references:
  style: "RSC numbered (Vancouver), superscript"
  csl: "paper/dd/royal-society-of-chemistry.csl"
  notes: "Superscript numbers after punctuation, in order of first citation (incl. tables and captions); all author names; CASSI abbreviations; references cited only in the ESI listed at the end of the main list with a note in the DAS."

sections:
  order:
    - Abstract
    - Introduction
    - Conclusions
    - Author contributions
    - Conflicts of interest
    - Data availability
    - Acknowledgements
    - Notes and references
  required:
    - Abstract
    - Conflicts of interest
    - Data availability
    - Acknowledgements
    - Notes and references

figures:
  formats: [tiff]
  min_dpi: 600
  naming: "Fig{n}.tif"
  placement: inline

anonymize: false

files:
  - manuscript
  - esi
  - cover_letter
---

# RSC Advances submission spec (xrd-clip)

Source: RSC Advances author guidelines, https://www.rsc.org/publishing/publish-with-us/publish-a-journal-article/rsc-advances
("Prepare your article" and "Submit your article" tabs), read in Chrome on 2026-10-18 (rsc.org blocks scripted fetches).
Journal page: https://www.rsc.org/publishing/journals/rsc-advances (APC, scope).

## Traceability

- Article type Paper, no length limit: "Although there is no page limit for papers, appropriateness of length to content of new science will be taken into consideration."
- Scope: "Papers that contain little or no chemistry and are not considered to be of interest or relevance to the chemistry community are not within the scope of the journal." -> the cover letter must make the chemistry case (organic molecular crystals, polymorph identification).
- Template optional, typeset by RSC: "Use of the template is optional for our journals." / "the manuscript will be formatted for you." -> page, font, spacing, line numbers and page numbers are NOT journal rules; they are reviewer-convenience choices (user decision pending, see Ambiguities 1).
- Abstract: "The abstract is a single paragraph which should: be around 50 to 250 words" -> abstract_words 250 (current: 250).
- Keywords: the guidelines ask for none -> keywords null.
- Section order: "A data availability statement must be included at the end of the article under the heading "Data availability", after the conflicts of interest statement and before any acknowledgements." -> current order (Data availability before Author contributions) must change.
- Conflicts: "This statement is required for all submitted manuscripts. If no conflicts exist, please state that 'There are no conflicts to declare'." (present)
- Author contributions: "we strongly encourage ... We strongly recommend you use CRediT" (present, not in CRediT terms yet; see Ambiguities 4).
- Acknowledgements: "You should also declare all sources of funding and any use of artificial intelligence (AI) tools at this point." -> AI-use statement required here (placeholder today).
- References: "It's important you use Vancouver style (not Harvard style)." / "Use superscript numbers ... at the end of the sentence (after the punctuation)" / "The names and initials of all authors should be given" / "Chemical Abstracts Service Source Index (CASSI)". The repo's royal-society-of-chemistry.csl implements this.
- ESI references: "References cited in the Supplementary Information (SI) may be listed at the bottom of the reference section of the main manuscript. Please include a note in the data availability statement". (esi.md cites no bib keys today; Table S24 gives DOIs.)
- Figures: "supplied as TIFF files, with a resolution of 600 dpi or greater" / "single column (8.3 cm) or double column (17.1 cm) width ... no longer than 23.3 cm" -> figures.min_dpi 600, TIFF.
- TOC graphic: "A table of contents entry (graphical abstract) is required, which should be submitted at the revision stage." 8 cm x 4 cm, TIFF 600 dpi, text max 250 characters -> NOT needed now.
- DAS: "The following statement is generally not acceptable 'Data are available upon request from the authors'." / "any data required to understand and verify the research in an article must be made available on submission." / "The full URL link to data sets should be provided (not embedded behind text)." -> the Zenodo deposit must exist BEFORE submission.
- Cover letter: "Address your letter to the relevant Associate Editor or Executive Editor." / "Make sure you state the correct journal name." / "Don't include preferred/non-preferred reviewers in your letter as these should be entered in the manuscript submission system only." / "Your cover letter will be sent to reviewers."
- Submission: "Articles should be submitted using ScholarOne Manuscripts". Frank submits manually.
- APC: "This journal charges an APC of £2,200." Purdue's RSC open-access agreement covers all RSC journals to 2026-12-31, applied at acceptance with the Purdue e-mail (memory: mlip-dynstab, Purdue Libraries); first decision (peer reviewed) 27 days.

## File inventory

1. Manuscript .docx (figures embedded; TIFF originals at 600 dpi supplied at revision or on request).
2. ESI .docx (or PDF).
3. Cover letter (ScholarOne Step 1 box or upload).
4. In ScholarOne only: suggested/non-preferred reviewers.
5. NOT now: TOC graphic + text (revision stage).

## Ambiguities

1. Layout (not mandated): A4, 2.5 cm margins, Times New Roman 12 pt, 1.5 spacing, continuous line numbers, page numbers. Recommended for reviewers.
2. Zenodo deposit + GitHub URL: the DAS must point to a live deposit at submission. The repo is private; deposit to Zenodo (public) without flipping the GitHub repo, or flip it too?
3. AI-use statement: must be in Frank's own words (memory rule); placeholder today.
4. Author contributions in CRediT terms (sole author): "Conceptualization, Methodology, Software, Formal analysis, Investigation, Data curation, Visualization, Writing - original draft, Writing - review & editing."
5. Kaduk papers: cite the 18 reports in the reference list (needed for attribution in Section 4.7) vs DOIs in Table S24 only.
6. Acknowledge ICDD (Nicole) for the deposited donepezil CIF?
7. Cover letter addressee: no named editor for a new submission -> "Dear Editor" / "Dear Associate Editor".
8. Figures as 600-dpi TIFF now (current PNGs are 300 dpi) or at revision.

"""Post-format fixes after the journal-submission skill's apply_format.py:
the Title style keeps pandoc's 28 pt (set 16 pt bold black) and the Hyperlink style is blue (set black, no underline),
so citations and reference entries print black. Usage: python postfix.py <manuscript.docx>"""
import sys

from docx import Document
from docx.shared import Pt, RGBColor

path = sys.argv[1]
doc = Document(path)
title = doc.styles["Title"]
title.font.size = Pt(16)
title.font.bold = True
title.font.name = "Times New Roman"
title.font.color.rgb = RGBColor(0, 0, 0)
link = doc.styles["Hyperlink"]
link.font.color.rgb = RGBColor(0, 0, 0)
link.font.underline = False
doc.save(path)
print("postfix ok:", path)

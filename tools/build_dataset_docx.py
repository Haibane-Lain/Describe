"""Build the short dataset description DOCX for the CS 403 intelligent-agent project.

Deliberately brief and non-technical: the group has selected and prepared the data,
and model training has not started. Counts were measured from the working copy of
the corpus under ``data/``.
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, Inches, RGBColor

from docx.oxml.ns import qn

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "Dataset-Description-VizWiz-Flickr30k.docx"
OUT = DEFAULT_OUT


def repeat_header(table):
    """Keep table rows whole, and repeat the header row if the table spans pages."""
    tr = table.rows[0]._tr
    trPr = tr.get_or_add_trPr()
    trPr.append(trPr.makeelement(qn("w:tblHeader"), {}))
    for row in table.rows:
        trPr = row._tr.get_or_add_trPr()
        trPr.append(trPr.makeelement(qn("w:cantSplit"), {}))


def body(doc, text, italic=False, space_after=8):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.add_run(text).italic = italic
    return p


def bullets(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(3)
        if isinstance(item, tuple):
            lead, rest = item
            p.add_run(lead).bold = True
            p.add_run(rest)
        else:
            p.add_run(item)


def table(doc, header, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, text in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(text)
        run.bold = True
        run.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for i, text in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(text))
            run.font.size = Pt(9)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    return t


def main() -> None:
    doc = Document()

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(8)

    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.9)
        section.right_margin = Inches(0.9)

    doc.add_heading("Dataset Description", level=0)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Technological Institute of the Philippines — Quezon City\n"
                  "College of Computer Studies — Computer Science Department\n"
                  "CS 403 — Intelligent Agents | First Semester, S.Y. 2026–2027")
    r.font.size = Pt(10)
    r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Mirandilla, Chesterfield S. · Natividad, Antonio Joaquin D. · "
                  "Navarro, Achaiah Andrei V. · Pascual, Marc Isaac M.")
    r.font.size = Pt(9)
    r.italic = True
    p.paragraph_format.space_after = Pt(12)

    # ==================================================================== PART A
    doc.add_heading("Part A. Dataset Profile", level=1)

    doc.add_heading("Proposed System", level=2)
    body(doc, "An Intelligent Image Description and Text Reading System for Blind and "
              "Low-Vision Users. It is a mobile app that takes a photo, checks that the photo "
              "is clear enough to use, describes it out loud, and reads printed text aloud "
              "when the photo contains any.")

    doc.add_heading("Dataset Name", level=2)
    body(doc, "VizWiz + Flickr30k Assistive Vision Dataset — made up of VizWiz-Captions, "
              "VizWiz Image Quality Issues, Flickr30k, public text-reading datasets, and a "
              "small set of photos the group takes itself.")

    doc.add_heading("Dataset Source", level=2)
    bullets(doc, [
        ("VizWiz-Captions and VizWiz Image Quality Issues — ",
         "existing datasets (Gurari et al., 2020; Chiu et al., 2020), from the official "
         "VizWiz archives. Released under CC BY 4.0."),
        ("Flickr30k — ",
         "an existing dataset (Young et al., 2014), from Hugging Face. The images remain "
         "Flickr user content."),
        ("TextOCR and HierText — ",
         "existing datasets used only to check the text reading. The reading is done by "
         "EasyOCR, a ready-made tool the group does not train, so these photos are not "
         "training data."),
        ("Group photographs — ",
         "Created by the Group: ten to twenty photos of local products, medicine boxes, and "
         "documents, to check the app on everyday Philippine items."),
    ])

    doc.add_heading("Dataset Purpose", level=2)
    body(doc, "The dataset contains real photos taken by blind users, each with descriptions "
              "written in ordinary language, plus people's judgments on whether a photo is "
              "too poor to describe and what is wrong with it. This is what lets the app "
              "notice a bad photo and ask for another one instead of guessing. A note saying "
              "which photos contain text, together with public text photos whose words are "
              "written out, supports reading printed text aloud.")

    doc.add_heading("Dataset Size", level=2)
    body(doc, "The main corpus holds 71,487 photos, over 310,000 descriptions, and 196,065 "
              "quality judgments, about 40 GB in all. The text-reading datasets add about "
              "40,000 more photos with the words written out.")

    table(doc,
          ["Part", "Photos", "What it gives"],
          [
              ["Flickr30k", "31,783", "5 descriptions per photo"],
              ["VizWiz (captions and quality)", "39,181",
               "descriptions, quality judgments, text-present note"],
              ["TextOCR and HierText", "about 40,000", "the words in the photo, written out"],
              ["Group photographs", "10–20", "local products and medicine boxes"],
          ],
          widths=[1.9, 1.1, 3.7])
    repeat_header(doc.tables[-1])

    body(doc, "The VizWiz photos are counted once, since the caption and quality data "
              "describe the same 39,181 photos. Only the VizWiz and Flickr30k counts were "
              "counted from the files downloaded so far.", italic=True, space_after=12)

    # ==================================================================== PART B
    doc.add_heading("Part B. Dataset Questions", level=1)

    doc.add_heading("1. What problem does this dataset support?", level=2)
    body(doc, "The main problem is that a dark, blurry, or tilted photo can make the app say "
              "something wrong, and the user cannot see the photo to check it. The dataset "
              "helps because it contains real photos taken by blind users, including many "
              "that are too poor to describe, together with people's judgments saying so. The "
              "app can learn to spot these photos and ask for another one. The descriptions "
              "and the text photos support the other two answers the app gives: describing a "
              "scene, and reading text aloud.")

    doc.add_heading("2. What are the most important elements of the dataset?", level=2)
    bullets(doc, [
        ("Quality judgments. ",
         "Five people looked at each photo and said whether it could be described at all and "
         "what was wrong with it — blur, darkness, brightness, tilt, something blocking the "
         "view, or poor framing. This teaches the app when to ask for another photo and what "
         "to do about each problem."),
        ("A note on whether text is present. ",
         "About six in ten photos contain visible text, such as a label or a document. This "
         "tells the app whether to describe the scene or read the text aloud."),
        ("Descriptions of each photo. ",
         "Most photos come with several descriptions written in everyday language, which is "
         "the wording the app learns to imitate. A few photos have no description because "
         "the quality was too poor — useful examples of photos that should never be "
         "described."),
    ])

    doc.add_heading("3. How can the agent use this information?", level=2)
    body(doc, "Before answering, the app uses the quality judgments to decide whether the "
              "photo is usable. If it is not, the app asks for another photo instead of "
              "describing it. The notes on what is wrong tell it what to do: ask for a retake "
              "when the photo is blurry or dark, straighten a tilted photo automatically, or "
              "describe what is visible with a warning when the view is blocked. The "
              "descriptions teach it how to describe a scene, and the text photos are used to "
              "check that printed text is read correctly and to decide when the app should "
              "warn the user that it is unsure.")

    doc.add_heading("4. Is the dataset sufficient?", level=2)
    body(doc, "Yes, for all three parts. There are tens of thousands of real photos with "
              "descriptions and quality judgments, far more than the group could collect on "
              "its own, and the text photos give the correct words so that reading can be "
              "checked. Only a few hundred text photos are needed, so one public test set "
              "plus a small set of the group's own photos is enough. Two things are still "
              "missing: the photos are mostly American and European, so local products, "
              "medicine labels, and peso bills are covered only by the group's few photos, "
              "and there is no feedback yet from blind or low-vision users, which only a user "
              "study can provide.")

    doc.add_heading("5. What are the dataset's limitations?", level=2)
    bullets(doc, [
        ("The judgments are opinions. ",
         "Five people rated each photo and the majority decided, so “usable” and “unusable” "
         "are approximate rather than exact."),
        ("Very poor photos are few. ",
         "Only about one in ten photos is unusable, and some problems, such as a blocked "
         "view or a photo that is too bright, appear in very few photos, so the app may "
         "handle these cases poorly."),
        ("The descriptions are uneven. ",
         "Some photos have fewer descriptions and a few have none, and the writing has "
         "capitals and typos because it comes from real users. The public text photos are "
         "also clear and well framed, so they make reading look easier than a shaky photo of "
         "a medicine box."),
        ("Privacy and consent matter. ",
         "The photos show real people's belongings, documents, and labels and were given by "
         "blind users, so they must not be shared freely. VizWiz is under CC BY 4.0 and "
         "Flickr30k has its own terms; any photos the group takes need the person's consent."),
    ])

    # =================================================================== sources
    doc.add_heading("Sources", level=2)
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(
        "Chiu, T.-Y., Zhao, Y., & Gurari, D. (2020). Assessing image quality issues for "
        "real-world problems. CVPR. · Gurari, D., Zhao, Y., Zhang, M., & Bhattacharya, N. "
        "(2020). Captioning images taken by people who are blind. ECCV. · Young, P., Lai, A., "
        "Hodosh, M., & Hockenmaier, J. (2014). From image descriptions to visual denotations. "
        "TACL, 2, 67–78. · Singh, A., et al. (2021). TextOCR. CVPR. · Long, S., et al. "
        "(2022). HierText. CVPR.")
    r.font.size = Pt(9)
    p = doc.add_paragraph()
    r = p.add_run("VizWiz: vizwiz.cs.colorado.edu/VizWiz_final · Flickr30k: Hugging Face "
                  "lmms-lab-encoder/flickr30k")
    r.font.size = Pt(9)
    r.italic = True

    doc.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

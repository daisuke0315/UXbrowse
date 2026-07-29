from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUT_DIR = Path(__file__).resolve().parent
MANUSCRIPT_PATH = OUT_DIR / "Medical_Physics_Blinded_Manuscript_Draft.docx"
TITLE_PAGE_PATH = OUT_DIR / "Medical_Physics_Title_Page_Draft.docx"

FONT = "Arial"
INK = RGBColor(0, 0, 0)
PLACEHOLDER_FILL = "FFF2CC"
TABLE_HEADER_FILL = "E7E6E6"


def set_rfont_name(rfonts, name):
    for attr in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        key = qn(f"w:{attr}")
        if key in rfonts.attrib:
            del rfonts.attrib[key]
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)
    rfonts.set(qn("w:eastAsia"), name)


def set_run_font(run, name=None, size=11, bold=None, italic=None, color=INK):
    if name is None:
        name = FONT
    run.font.name = name
    run._element.get_or_add_rPr()
    set_rfont_name(run._element.rPr.rFonts, name)
    run.font.size = Pt(size)
    run.font.color.rgb = color
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths):
    assert sum(widths) == 9360
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.autofit = False
    tbl_pr = table._tbl.tblPr

    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), "9360")
    tbl_w.set(qn("w:type"), "dxa")

    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), "120")
    tbl_ind.set(qn("w:type"), "dxa")

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)

    for row in table.rows:
        row.height = None
        for idx, cell in enumerate(row.cells):
            cell.width = Inches(widths[idx] / 1440)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(widths[idx]))
            tc_w.set(qn("w:type"), "dxa")
            set_cell_margins(cell)


def mark_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = tr_pr.find(qn("w:tblHeader"))
    if tbl_header is None:
        tbl_header = OxmlElement("w:tblHeader")
        tr_pr.append(tbl_header)
    tbl_header.set(qn("w:val"), "true")


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.extend([fld_begin, instr, fld_sep, text, fld_end])
    set_run_font(run, size=9)


def set_line_numbering(section):
    sect_pr = section._sectPr
    existing = sect_pr.find(qn("w:lnNumType"))
    if existing is not None:
        sect_pr.remove(existing)
    ln = OxmlElement("w:lnNumType")
    ln.set(qn("w:countBy"), "1")
    ln.set(qn("w:start"), "1")
    ln.set(qn("w:restart"), "continuous")
    ln.set(qn("w:distance"), "360")
    sect_pr.append(ln)


def configure_document(doc, line_numbers=False):
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)
    if line_numbers:
        set_line_numbering(section)

    normal = doc.styles["Normal"]
    normal.font.name = FONT
    set_rfont_name(normal._element.rPr.rFonts, FONT)
    normal.font.size = Pt(11)
    normal.font.color.rgb = INK
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.first_line_indent = Inches(0.25)

    for style_name, size, before, after in (
        ("Heading 1", 13, 12, 4),
        ("Heading 2", 11, 10, 2),
        ("Heading 3", 11, 8, 2),
    ):
        style = doc.styles[style_name]
        style.font.name = FONT
        set_rfont_name(style._element.rPr.rFonts, FONT)
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = INK
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.line_spacing = 1.5
        style.paragraph_format.first_line_indent = Inches(0)
        style.paragraph_format.keep_with_next = True

    caption = doc.styles["Caption"]
    caption.font.name = FONT
    set_rfont_name(caption._element.rPr.rFonts, FONT)
    caption.font.size = Pt(10)
    caption.font.color.rgb = INK
    caption.font.bold = False
    caption.paragraph_format.line_spacing = 1.5
    caption.paragraph_format.space_before = Pt(4)
    caption.paragraph_format.space_after = Pt(4)
    caption.paragraph_format.first_line_indent = Inches(0)

    footer = section.footer
    footer_p = footer.paragraphs[0]
    add_page_number(footer_p)


def shade_placeholder(run):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), PLACEHOLDER_FILL)
    run._element.get_or_add_rPr().append(shd)


def add_rich_paragraph(doc, segments, style=None, indent=True, align=None):
    p = doc.add_paragraph(style=style)
    if not indent:
        p.paragraph_format.first_line_indent = Inches(0)
    if align is not None:
        p.alignment = align
    for segment in segments:
        if isinstance(segment, str):
            segment = {"text": segment}
        run = p.add_run(segment["text"])
        set_run_font(
            run,
            size=segment.get("size", 11),
            bold=segment.get("bold"),
            italic=segment.get("italic"),
        )
        if segment.get("sup"):
            run.font.superscript = True
        if segment.get("placeholder"):
            shade_placeholder(run)
    return p


def add_plain(doc, text, style=None, indent=True):
    return add_rich_paragraph(doc, [{"text": text}], style=style, indent=indent)


def add_heading(doc, text, level=1):
    return add_plain(doc, text, style=f"Heading {level}", indent=False)


def add_abstract_label(doc, label, text):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Inches(0)
    p.paragraph_format.line_spacing = 1.5
    r1 = p.add_run(f"{label}: ")
    set_run_font(r1, bold=True)
    r2 = p.add_run(text)
    set_run_font(r2)
    return p


def add_placeholder_paragraph(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Inches(0)
    r = p.add_run(f"[TO COMPLETE: {text}]")
    set_run_font(r, bold=True)
    shade_placeholder(r)
    return p


def fill_table_cell(
    cell,
    text,
    bold=False,
    align=WD_ALIGN_PARAGRAPH.LEFT,
    placeholder=False,
    size=9.5,
):
    p = cell.paragraphs[0]
    p.paragraph_format.first_line_indent = Inches(0)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.15
    p.alignment = align
    r = p.add_run(text)
    set_run_font(r, size=size, bold=bold)
    if placeholder:
        shade_placeholder(r)


def add_reference(doc, text):
    p = doc.add_paragraph(style="List Number")
    p.paragraph_format.first_line_indent = Inches(0)
    p.paragraph_format.left_indent = Inches(0.35)
    p.paragraph_format.right_indent = Inches(0)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    r = p.add_run(text)
    set_run_font(r, size=10)


def add_title_page():
    doc = Document()
    configure_document(doc, line_numbers=False)
    doc.styles["Normal"].paragraph_format.line_spacing = 1.15
    doc.styles["Normal"].paragraph_format.first_line_indent = Inches(0)
    for style_name in ("Heading 1", "Heading 2", "Heading 3"):
        doc.styles[style_name].paragraph_format.line_spacing = 1.15
        doc.styles[style_name].paragraph_format.space_before = Pt(7)
        doc.styles[style_name].paragraph_format.space_after = Pt(2)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Inches(0)
    p.paragraph_format.space_after = Pt(12)
    r = p.add_run(
        "CT-coordinate-constrained optical three-dimensional reconstruction with "
        "view-driven multiplanar CT display: a phantom accuracy study"
    )
    set_run_font(r, size=15, bold=True)

    add_rich_paragraph(
        doc,
        [{"text": "Article type: ", "bold": True}, {"text": "Research Article"}],
        indent=False,
    )
    add_rich_paragraph(
        doc,
        [{"text": "Running title: ", "bold": True}, {"text": "CT-constrained optical 3D and view-driven MPR"}],
        indent=False,
    )

    add_heading(doc, "Authors", 1)
    add_rich_paragraph(
        doc,
        [
            {"text": "Daisuke Tanii", "bold": True},
            {"text": " [CONFIRM ROMANIZATION AND DEGREES]", "placeholder": True},
            {"text": "; [COAUTHOR FULL NAMES, DEGREES]", "placeholder": True},
        ],
        indent=False,
    )

    add_heading(doc, "Affiliations", 1)
    add_rich_paragraph(
        doc,
        [
            {"text": "1. SAFRA, Sapporo Konjinkai Memorial Hospital, Sapporo, Japan"},
            {"text": " [CONFIRM OFFICIAL ENGLISH AFFILIATION]", "placeholder": True},
        ],
        indent=False,
    )
    add_placeholder_paragraph(doc, "Add numbered affiliations for all coauthors.")

    add_heading(doc, "Corresponding author", 1)
    add_placeholder_paragraph(
        doc,
        "Full name, department, institution, postal address, country, telephone number, and email address.",
    )

    add_heading(doc, "Author contributions", 1)
    add_plain(
        doc,
        "D.T.: conceptualization, methodology, software, investigation, data curation, formal analysis, "
        "visualization, and writing—original draft. [COAUTHORS]: validation, clinical interpretation, "
        "supervision, and writing—review and editing.",
        indent=False,
    )
    add_placeholder_paragraph(doc, "Confirm CRediT roles and obtain approval from every author.")

    add_heading(doc, "Acknowledgments", 1)
    add_placeholder_paragraph(doc, "List nonauthor contributions, or state “None.”")

    add_heading(doc, "Funding", 1)
    add_plain(
        doc,
        "This work received no external funding and was conducted using the authors’ own resources and "
        "existing institutional infrastructure.",
        indent=False,
    )
    add_placeholder_paragraph(doc, "Confirm the funding statement before submission.")

    add_heading(doc, "Conflict of interest", 1)
    add_plain(
        doc,
        "D.T. is an inventor and applicant on a Patent Cooperation Treaty application related to the "
        "integration of optical three-dimensional models with internal medical imaging and may have a "
        "future financial interest in commercialization or external collaboration. The study was designed "
        "and evaluated without support from a specific commercial entity. The remaining authors declare "
        "no relevant conflicts of interest.",
        indent=False,
    )
    add_placeholder_paragraph(
        doc,
        "Insert the PCT application number and confirm the complete disclosure with the institution and all authors.",
    )

    add_heading(doc, "Manuscript declarations", 1)
    add_plain(
        doc,
        "The corresponding author confirms that all authors have reviewed and approved the manuscript and "
        "that the work is not under consideration elsewhere.",
        indent=False,
    )

    doc.core_properties.title = "Medical Physics title page draft"
    doc.core_properties.author = ""
    doc.core_properties.last_modified_by = ""
    doc.save(TITLE_PAGE_PATH)


def add_main_manuscript():
    doc = Document()
    configure_document(doc, line_numbers=True)
    doc.core_properties.title = "Medical Physics blinded manuscript draft"
    doc.core_properties.author = ""
    doc.core_properties.last_modified_by = ""
    doc.core_properties.comments = "Working draft with highlighted completion items."

    header = doc.sections[0].header
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    hr = hp.add_run("DRAFT — NOT FOR SUBMISSION")
    set_run_font(hr, size=9, bold=True)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.first_line_indent = Inches(0)
    title.paragraph_format.space_after = Pt(10)
    r = title.add_run(
        "CT-coordinate-constrained optical three-dimensional reconstruction with "
        "view-driven multiplanar CT display: a phantom accuracy study"
    )
    set_run_font(r, size=15, bold=True)

    add_heading(doc, "Abstract", 1)
    add_abstract_label(
        doc,
        "Background",
        "Computed tomography (CT) preserves internal anatomy in a physical patient coordinate system but "
        "does not directly depict the patient-specific visible surface. Optical three-dimensional (3D) "
        "reconstruction preserves surface appearance but ordinarily lacks absolute scale and correspondence "
        "with Digital Imaging and Communications in Medicine (DICOM) coordinates.",
    )
    add_abstract_label(
        doc,
        "Purpose",
        "To develop and technically evaluate an optical 3D reconstruction constrained to the CT coordinate "
        "system during bundle adjustment and integrated with multiplanar reconstruction (MPR) driven by the "
        "observer’s virtual viewpoint.",
    )
    add_abstract_label(
        doc,
        "Methods",
        "Fiducial markers visible in CT and optical images were placed around a phantom. Marker coordinates "
        "measured in DICOM physical space were introduced as ground control point (GCP) constraints during "
        "structure-from-motion and multiview-stereo reconstruction. A viewing plane was defined from the "
        "virtual camera position, target, and up vector, and CT values were resampled in physical coordinates "
        "to generate a view-driven MPR image. Geometric shape agreement was evaluated at four points spanning "
        "approximately 40 cm in three reconstructions (12 observations). End-to-end target registration error "
        "(TRE) was measured at four non-GCP landmarks spanning approximately 30 cm in three models under GCP-"
        "constrained and unconstrained conditions (12 observations per condition).",
    )
    add_abstract_label(
        doc,
        "Results",
        "The post-alignment 3D geometric point error was 0.27 ± 0.12 mm. End-to-end TRE was 0.45 ± 0.26 mm "
        "with GCP constraints and 1.47 ± 0.95 mm without GCP constraints. The constrained reconstruction "
        "reduced the mean TRE by 1.02 mm (69.4%).",
    )
    add_abstract_label(
        doc,
        "Conclusions",
        "Introducing CT-derived GCPs during optical reconstruction enabled submillimeter integration of an "
        "optical 3D model and view-driven CT MPR under static phantom conditions. The geometric result after "
        "Procrustes and iterative closest point alignment represents shape agreement rather than uncorrected "
        "absolute coordinate accuracy. Larger experiments are required to characterize repeatability, "
        "uncertainty, motion sensitivity, and performance in human subjects.",
    )

    add_rich_paragraph(
        doc,
        [
            {"text": "Keywords: ", "bold": True},
            {"text": "computed tomography; optical three-dimensional reconstruction; ground control point; "
                     "bundle adjustment; image registration; multiplanar reconstruction"},
        ],
        indent=False,
    )

    add_heading(doc, "1. Introduction", 1)
    add_rich_paragraph(
        doc,
        [
            {"text": "Computed tomography (CT) provides high-resolution internal anatomical information in a "
                     "physical patient coordinate system and is central to diagnosis, treatment planning, and "
                     "assessment of therapeutic response. Conventional cross-sectional and volume-rendered CT "
                     "displays, however, do not retain the patient’s visible-light appearance, including skin "
                     "color, external landmarks, and surface texture. Consequently, the relationship between an "
                     "internal target and its location on the visible body surface may remain difficult to "
                     "communicate to nonexpert observers."},
            {"text": "1,2", "sup": True},
        ],
    )
    add_rich_paragraph(
        doc,
        [
            {"text": "Photogrammetric structure from motion (SfM), multiview stereo (MVS), neural radiance "
                     "fields, and 3D Gaussian splatting can reconstruct realistic 3D appearance from optical "
                     "images."},
            {"text": "3,4", "sup": True},
            {"text": " An optical reconstruction derived solely from image correspondences is typically defined "
                     "up to a similarity transformation and therefore does not inherently share the scale, "
                     "orientation, origin, or patient coordinates of the associated CT dataset. A subsequent "
                     "rigid or similarity registration can align a completed surface model with CT, but it does "
                     "not constrain scale drift, camera poses, or local model deformation during reconstruction."},
            {"text": "4,5", "sup": True},
        ],
    )
    add_rich_paragraph(
        doc,
        [
            {"text": "Ground control points (GCPs) provide known 3D coordinates that can be incorporated into "
                     "bundle adjustment. Their use is established in photogrammetry, where their number and "
                     "spatial distribution affect global accuracy."},
            {"text": "5", "sup": True},
            {"text": " In medical imaging, registration of optical surface models with CT or cone-beam CT has "
                     "also been investigated, including high-accuracy applications in dental anatomy."},
            {"text": "6", "sup": True},
            {"text": " A remaining technical challenge is to generate a patient-specific visible-light model "
                     "directly in DICOM physical space and to link the observer’s view of that model to the "
                     "definition of an arbitrary CT section."},
        ],
    )
    add_plain(
        doc,
        "The purpose of this study was to develop a system that (1) incorporates CT-derived GCP coordinates "
        "during optical bundle adjustment, (2) constructs the visible-light 3D model in the CT physical "
        "coordinate system, and (3) generates an MPR plane from the observer’s virtual camera pose. We "
        "performed a static phantom study to quantify post-alignment geometric agreement and end-to-end "
        "display TRE, including a comparison with an otherwise equivalent reconstruction produced without "
        "GCP constraints.",
    )

    add_heading(doc, "2. Materials and methods", 1)
    add_heading(doc, "2.1 System architecture", 2)
    add_plain(
        doc,
        "The system comprised four stages: acquisition of CT and optical images; GCP-constrained optical 3D "
        "reconstruction; generation of an arbitrary MPR image in DICOM physical coordinates; and integrated "
        "rendering of the optical model and MPR plane. The principal hardware and software components are "
        "summarized in Table I. Figure 1 shows the complete data flow. CT-derived marker coordinates and the "
        "optical reconstruction were expressed in a common right-handed physical coordinate system. "
        "Coordinate-axis conventions, transformation matrices, and unit conversions were explicitly retained "
        "throughout the processing pipeline.",
    )
    add_placeholder_paragraph(
        doc,
        "Confirm the coordinate handedness, patient orientation convention, transformation order, and whether any axis permutation or sign inversion was applied.",
    )

    add_heading(doc, "2.2 Phantom and fiducial configuration", 2)
    add_plain(
        doc,
        "A static phantom was surrounded by markers that could be localized in both CT and visible-light "
        "images. A subset of markers was designated as GCPs for reconstruction, whereas separate landmarks "
        "were reserved for accuracy assessment. Marker centers were identified in CT physical coordinates "
        "and in the reconstructed optical model. The spatial arrangement was selected to distribute control "
        "and evaluation points over the reconstructed volume rather than concentrating them in a single "
        "region.",
    )
    add_placeholder_paragraph(
        doc,
        "Provide the phantom name, manufacturer, material, dimensions, marker material and diameter, total number of markers, number used as GCPs, number reserved for evaluation, and a diagram with intermarker distances.",
    )
    add_placeholder_paragraph(
        doc,
        "Confirm whether the four points in the geometric shape analysis were independent of the GCP set. If they were GCPs, revise the analysis label and interpretation.",
    )

    add_heading(doc, "2.3 CT acquisition and landmark localization", 2)
    add_plain(
        doc,
        "The phantom was imaged using a clinical CT system. DICOM metadata, including Image Position "
        "(Patient), Image Orientation (Patient), Pixel Spacing, and slice positions, were retained. The "
        "physical coordinate of each marker center was determined from the CT volume. CT values at arbitrary "
        "physical coordinates were later sampled by trilinear interpolation for MPR generation.",
    )
    add_placeholder_paragraph(
        doc,
        "Add CT manufacturer and model, acquisition mode, tube potential (kV), tube current or exposure, pitch, collimation, reconstructed slice thickness and interval, field of view, matrix, reconstruction kernel, and voxel dimensions.",
    )
    add_placeholder_paragraph(
        doc,
        "Describe the marker-center localization method, observer experience, number of repeated measurements, and intraobserver/interobserver localization uncertainty.",
    )

    add_heading(doc, "2.4 Optical acquisition and three-dimensional reconstruction", 2)
    add_plain(
        doc,
        "Visible-light video was acquired with an iPhone 15 Pro (Apple Inc., Cupertino, CA, USA) while the "
        "camera was moved around the stationary phantom. Frames were extracted using FFmpeg and processed "
        "with an SfM/MVS workflow. Camera intrinsics, camera poses, and sparse and dense 3D structure were "
        "estimated. GCP image observations were linked to the corresponding DICOM physical coordinates before "
        "the final bundle adjustment. RealityCapture was used for SfM/MVS reconstruction, and PostShot was "
        "used when a 3D Gaussian splatting representation was generated.",
    )
    add_placeholder_paragraph(
        doc,
        "Add video resolution, frame rate, exposure/focus settings, acquisition distance, trajectory, lighting conditions, number of extracted frames, frame-selection rule, and all software versions and reconstruction settings.",
    )
    add_placeholder_paragraph(
        doc,
        "State whether accuracy measurements were made on a mesh, point cloud, or Gaussian representation and describe how the marker center was selected in that representation.",
    )

    add_heading(doc, "2.5 CT-coordinate-constrained bundle adjustment", 2)
    add_plain(
        doc,
        "Conventional bundle adjustment minimizes the reprojection difference between an observed image "
        "location and the projection of an estimated 3D point. In the constrained reconstruction, a second "
        "term penalized the difference between reconstructed GCP locations and their known CT coordinates:",
    )
    eq = doc.add_paragraph()
    eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
    eq.paragraph_format.first_line_indent = Inches(0)
    eq.paragraph_format.space_before = Pt(6)
    eq.paragraph_format.space_after = Pt(6)
    er = eq.add_run(
        "min Σᵢⱼ ‖π(Kᵢ, Rᵢ, tᵢ, Xⱼ) − xᵢⱼ‖² + λ Σₖ ‖Xgₖ − Gₖ‖²,"
    )
    set_run_font(er, name="Cambria Math", size=11)
    add_plain(
        doc,
        "where xᵢⱼ is the observed image position of feature j in image i; Xⱼ is the estimated 3D point; "
        "Kᵢ, Rᵢ, and tᵢ are the intrinsic matrix, rotation, and translation for camera i; Gₖ is the known "
        "CT coordinate of GCP k; Xgₖ is its reconstructed coordinate; π denotes projection; and λ denotes the "
        "relative weighting of the coordinate constraint. The GCP term constrained translation, rotation, "
        "and scale during reconstruction. An unconstrained model was generated from the same optical images "
        "without the GCP term.",
    )
    add_placeholder_paragraph(
        doc,
        "Report the GCP weighting or coordinate accuracy assigned in the software, residual thresholds, outlier rejection rule, final reprojection error, GCP residuals, and alignment method used for the unconstrained comparison.",
    )

    add_heading(doc, "2.6 View-driven MPR generation", 2)
    add_plain(
        doc,
        "The virtual camera position was denoted by eye, its look-at point by target, and its screen-up "
        "direction by up. The normal vector n and orthonormal in-plane axes u and v were calculated as",
    )
    for equation in (
        "n = normalize(target − eye),",
        "u = normalize(up × n),",
        "v = n × u.",
    ):
        eq = doc.add_paragraph()
        eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
        eq.paragraph_format.first_line_indent = Inches(0)
        rr = eq.add_run(equation)
        set_run_font(rr, name="Cambria Math", size=11)
    add_plain(
        doc,
        "For an MPR plane centered at center, the physical coordinate corresponding to pixel (i, j) was",
    )
    eq = doc.add_paragraph()
    eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
    eq.paragraph_format.first_line_indent = Inches(0)
    rr = eq.add_run(
        "X(i, j) = center + (i − cᵢ)Δu·u + (j − cⱼ)Δv·v,"
    )
    set_run_font(rr, name="Cambria Math", size=11)
    add_plain(
        doc,
        "where cᵢ and cⱼ are the image-center indices and Δu and Δv are the in-plane sampling intervals. "
        "Physical coordinates were transformed to continuous voxel indices using DICOM geometry, and CT "
        "values were obtained by trilinear interpolation. The eye, target, and up vectors were transmitted "
        "from the display application to the reconstruction service through a WebSocket connection. The "
        "generated MPR image was returned and mapped to the corresponding plane in the 3D scene.",
    )
    add_placeholder_paragraph(
        doc,
        "Add MPR matrix size, pixel spacing, field of view, interpolation implementation, CT window/level, update rate, latency measurement method, and handling of samples outside the CT volume.",
    )

    add_heading(doc, "2.7 Integrated rendering", 2)
    add_plain(
        doc,
        "The optical 3D model and CT volume were maintained in the same physical coordinate system. The MPR "
        "image was rendered as a texture on its corresponding plane. Changes in the virtual viewing direction "
        "updated the MPR plane definition, thereby linking navigation of the external model to navigation of "
        "the internal CT volume. FastAPI and WebSocket were used for communication, Python with SimpleITK, "
        "NumPy, and SciPy was used for DICOM processing and numerical calculations, and Unity was used for "
        "rendering and user-interface control.",
    )
    add_placeholder_paragraph(
        doc,
        "Add software and engine versions, workstation hardware, graphics hardware, operating systems, and the exact transformation applied when importing the reconstructed model into Unity.",
    )

    add_heading(doc, "2.8 Accuracy experiments", 2)
    add_heading(doc, "2.8.1 Post-alignment geometric shape agreement", 3)
    add_plain(
        doc,
        "Four corresponding points distributed over approximately 40 cm were used to evaluate the geometric "
        "shape of the optical reconstruction. Three models were generated, producing 12 point observations. "
        "To remove global translation, rotation, and scale before shape comparison, the point sets underwent "
        "centroid alignment, scale correction, singular-value-decomposition-based rotation estimation, and "
        "translation correction (Procrustes alignment), followed by iterative closest point refinement. The "
        "Euclidean distance between each paired point was calculated after alignment. Because global "
        "transformations were removed, this endpoint quantified post-alignment shape agreement and did not "
        "measure uncorrected CT-coordinate accuracy.",
    )
    add_placeholder_paragraph(
        doc,
        "Define an additional absolute-coordinate endpoint before Procrustes/ICP, or provide a clear rationale for limiting this experiment to shape agreement.",
    )

    add_heading(doc, "2.8.2 End-to-end integrated display accuracy", 3)
    add_plain(
        doc,
        "Four landmarks not used as GCPs and distributed over approximately 30 cm were used as independent "
        "targets. At each landmark, the 3D Euclidean distance between the marker center on the optical model "
        "and its corresponding location in the integrated MPR display was calculated as TRE. Three models "
        "were evaluated under the GCP-constrained and unconstrained conditions, yielding 12 observations per "
        "condition. The complete end-to-end measurement included optical acquisition, reconstruction, "
        "coordinate transfer, MPR generation, and integrated rendering.",
    )
    add_placeholder_paragraph(
        doc,
        "Describe the measurement tool, observer, blinding, repeated-point selection, voxel-to-display conversion, and any post hoc alignment applied in either condition. Confirm whether constrained and unconstrained observations are paired.",
    )

    add_heading(doc, "2.9 Statistical analysis", 2)
    add_plain(
        doc,
        "This initial technical feasibility experiment used descriptive statistics. Errors are reported as "
        "mean ± standard deviation (SD). The relative reduction in mean TRE was calculated as "
        "(mean TREunconstrained − mean TREconstrained)/mean TREunconstrained × 100%. No inferential test was "
        "performed because the study included three reconstructed models and the independence and pairing "
        "structure of individual point observations require confirmation from the raw data. Analyses were "
        "performed using Python.",
    )
    add_placeholder_paragraph(
        doc,
        "Insert individual observations, median, range, maximum error, 95% confidence intervals, and an analysis that accounts for repeated points nested within reconstruction runs. Add Python and package versions.",
    )

    add_heading(doc, "2.10 Ethical considerations", 2)
    add_plain(
        doc,
        "This technical study used a static anthropomorphic phantom and research image data only. It did not "
        "include human participants, patient-derived images, clinical information, biological specimens, or "
        "an intervention.",
    )
    add_placeholder_paragraph(
        doc,
        "Confirm with the institution whether a formal ethics determination or exemption statement is required and insert the reference number if applicable.",
    )

    add_heading(doc, "2.11 Generative artificial intelligence disclosure", 2)
    add_plain(
        doc,
        "OpenAI Codex was used to assist with English-language drafting, organization, and formatting of the "
        "manuscript. It was not used to generate experimental data, perform the reported analysis, or make "
        "scientific decisions. The authors reviewed and verified the manuscript and retain responsibility for "
        "all content.",
    )

    add_heading(doc, "3. Results", 1)
    add_heading(doc, "3.1 Reconstruction and system operation", 2)
    add_plain(
        doc,
        "The GCP-constrained workflow produced an optical 3D model in the CT physical coordinate system and "
        "enabled view-driven generation and placement of MPR images. A representative integrated display is "
        "shown in Figure 2.",
    )
    add_placeholder_paragraph(
        doc,
        "Report the number of attempted and successful reconstructions, failure criteria, processing time by stage, rendering frame rate, and MPR response latency.",
    )

    add_heading(doc, "3.2 Post-alignment geometric shape agreement", 2)
    add_plain(
        doc,
        "Across the three reconstructions and four points per reconstruction, the post-alignment 3D geometric "
        "point error was 0.27 ± 0.12 mm (12 observations). The evaluated points spanned approximately 40 cm. "
        "Figure 3 illustrates the point configuration and analysis sequence.",
    )
    add_placeholder_paragraph(
        doc,
        "Insert the individual 12 errors, median, range, maximum, and 95% confidence interval. Report the unaligned absolute-coordinate error separately.",
    )

    add_heading(doc, "3.3 End-to-end integrated display accuracy", 2)
    add_plain(
        doc,
        "The mean end-to-end TRE was 0.45 ± 0.26 mm under GCP-constrained reconstruction and 1.47 ± 0.95 mm "
        "under unconstrained reconstruction (12 observations per condition). The mean difference was 1.02 mm, "
        "corresponding to a 69.4% relative reduction. Results are summarized in Table II, and the planned "
        "distribution plot is shown schematically in Figure 4.",
    )
    add_placeholder_paragraph(
        doc,
        "Insert the raw paired data and replace the schematic distribution plot with the final dot/box plot. Report maximum errors and confidence intervals; use “significant” only if supported by a prespecified statistical test.",
    )

    add_heading(doc, "4. Discussion", 1)
    add_plain(
        doc,
        "This study demonstrated a technical workflow in which CT-derived physical coordinates were introduced "
        "during optical bundle adjustment and subsequently used to drive an integrated optical 3D and CT MPR "
        "display. Under static phantom conditions, end-to-end TRE averaged 0.45 mm with GCP constraints, "
        "compared with 1.47 mm without those constraints. The result supports the central technical premise "
        "that coordinate information incorporated during reconstruction can improve downstream alignment of "
        "an optical appearance model and a CT-derived display.",
    )
    add_plain(
        doc,
        "The approach differs from a workflow in which an unconstrained optical model is generated first and "
        "then registered to CT. Image-only bundle adjustment can produce a solution that is internally "
        "consistent in reprojection space without being expressed at the correct physical scale or in the "
        "patient coordinate system. The addition of GCP observations constrains the global similarity degrees "
        "of freedom during reconstruction and may also stabilize camera and point estimates. The reduction in "
        "end-to-end TRE observed here is consistent with this mechanism, although the small number of "
        "reconstruction runs precludes a precise estimate of the effect or its variability.",
    )
    add_plain(
        doc,
        "The two reported accuracy endpoints should be interpreted separately. The 0.27-mm result was measured "
        "after Procrustes and iterative closest point alignment, including scale correction. It therefore "
        "describes residual shape agreement after removal of the global coordinate transform. In contrast, the "
        "end-to-end TRE was intended to assess the complete integrated display at landmarks not used as GCPs. "
        "Reporting an additional uncorrected coordinate error, GCP residuals, and independent target errors "
        "will be necessary to establish the absolute accuracy of CT-coordinate-constrained reconstruction.",
    )
    add_plain(
        doc,
        "View-driven MPR provides a second distinction from conventional fixed axial, coronal, and sagittal "
        "navigation. The observer’s virtual camera pose directly defines the MPR basis, so the visible external "
        "region and the displayed internal section remain geometrically linked during navigation. This may be "
        "useful for image-guided planning, communication of internal-to-external spatial relationships, or "
        "education. These possible uses were not evaluated in the current experiment, and no claim regarding "
        "diagnostic performance, treatment guidance, patient understanding, or clinical safety can be made.",
    )
    add_plain(
        doc,
        "The reported errors are smaller than the scale of many routine visualization tasks, but a single "
        "accuracy threshold is not applicable to all medical uses. Acceptability depends on the intended use, "
        "target size, decision consequence, image resolution, motion, tissue deformation, and the availability "
        "of quality-control mechanisms. A system intended for patient explanation has different validation "
        "requirements from one intended for procedural navigation. Future experiments should therefore define "
        "the intended use before setting an acceptance criterion.",
    )

    add_heading(doc, "4.1 Limitations", 2)
    limitations = [
        "Only three reconstructed models and four evaluation landmarks per model were assessed.",
        "The observations at multiple landmarks within the same reconstruction are not necessarily statistically independent.",
        "The experiment used a static phantom and did not include respiration, body motion, soft-tissue deformation, or between-scan posture changes.",
        "The number, geometry, material, visibility, and localization uncertainty of the GCPs were not varied systematically.",
        "Post-alignment geometric analysis removed global scale and pose error and therefore did not measure absolute coordinate accuracy.",
        "Maximum error, confidence intervals, observer variability, device variability, and between-day repeatability were not available.",
        "The influence of the optical representation—mesh, point cloud, or view-dependent Gaussian model—on landmark localization was not quantified.",
        "Processing time, failure modes, quality-control thresholds, and clinical workflow were not formally evaluated.",
    ]
    for item in limitations:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.first_line_indent = Inches(0)
        p.paragraph_format.left_indent = Inches(0.5)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.5
        rr = p.add_run(item)
        set_run_font(rr)

    add_heading(doc, "4.2 Future validation", 2)
    add_plain(
        doc,
        "The next technical study should increase the number of independent reconstruction runs, report an "
        "uncertainty budget, and evaluate GCP number and spatial arrangement, marker-localization error, "
        "lighting, acquisition trajectory, and reconstruction settings. Constrained reconstruction should be "
        "compared with prespecified rigid and similarity registration baselines. Subsequent human-subject "
        "studies should separately test reproducibility, spatial understanding, task completion time, and "
        "clinical usability under an approved protocol.",
    )

    add_heading(doc, "5. Conclusions", 1)
    add_plain(
        doc,
        "CT-derived GCP constraints incorporated during optical bundle adjustment enabled a visible-light 3D "
        "model to be integrated with a view-driven CT MPR display. In a static phantom experiment, mean end-to-"
        "end TRE was 0.45 mm with GCP constraints and 1.47 mm without constraints. The findings establish "
        "technical feasibility but do not yet establish clinical performance. Absolute coordinate accuracy, "
        "repeatability, uncertainty, motion sensitivity, and human-subject performance require further study.",
    )

    add_heading(doc, "Data availability", 1)
    add_placeholder_paragraph(
        doc,
        "State where the minimum deidentified dataset—including individual landmark errors and acquisition metadata—will be deposited, or provide a justified controlled-access statement.",
    )

    add_heading(doc, "Code availability", 1)
    add_placeholder_paragraph(
        doc,
        "State which analysis and MPR-generation code can be shared, the repository and version/DOI, and any restrictions arising from commercial reconstruction software or patent considerations.",
    )

    add_heading(doc, "References", 1)
    references = [
        "Dappa E, Higashigaito K, Fornaro J, et al. Cinematic rendering—an alternative to volume rendering for 3D computed tomography imaging. Insights Imaging. 2016;7(6):849-856. doi:10.1007/s13244-016-0518-1.",
        "Pose-Díez-de-la-Lastra A, Ungi T, Morton D, et al. Real-time integration between Microsoft HoloLens 2 and 3D Slicer with demonstration in pedicle screw placement planning. Int J Comput Assist Radiol Surg. 2023;18:2023-2032. doi:10.1007/s11548-023-02977-0.",
        "Kerbl B, Kopanas G, Leimkühler T, Drettakis G. 3D Gaussian splatting for real-time radiance field rendering. ACM Trans Graph. 2023;42(4):139. doi:10.1145/3592433.",
        "Triggs B, McLauchlan PF, Hartley RI, Fitzgibbon AW. Bundle adjustment—a modern synthesis. In: Triggs B, Zisserman A, Szeliski R, eds. Vision Algorithms: Theory and Practice. Springer; 2000:298-372.",
        "Sanz-Ablanedo E, Chandler JH, Rodríguez-Pérez JR, Ordóñez C. Accuracy of unmanned aerial vehicle and SfM photogrammetry survey as a function of the number and location of ground control points used. Remote Sens. 2018;10(10):1606. doi:10.3390/rs10101606.",
        "Lim SW, et al. Registration accuracy between intraoral-scanned and cone-beam computed tomography-scanned crowns in various registration methods. Am J Orthod Dentofacial Orthop. 2020;157(3):348-356. doi:10.1016/j.ajodo.2019.04.031.",
        "Phelps EE, Wellings R, Griffiths F, et al. Do medical images aid understanding and recall of medical information? An experimental study comparing no image, a 2D medical image and a 3D medical image alongside a diagnosis. Patient Educ Couns. 2017;100(6):1120-1127.",
        "Stewart JA, et al. Visual teaching aids improve patient understanding and reduce anxiety prior to a colectomy. Am J Surg. 2021;222(4):780-785.",
        "Julious SA. Sample size of 12 per group rule of thumb for a pilot study. Pharm Stat. 2005;4(4):287-291. doi:10.1002/pst.185.",
    ]
    for ref in references:
        add_reference(doc, ref)
    add_placeholder_paragraph(
        doc,
        "Verify every reference against PubMed/Crossref; complete all author lists required by AMA style and confirm references 6 and 8 in particular.",
    )

    add_heading(doc, "Tables", 1)
    add_plain(doc, "Table I. System components and functions.", style="Caption", indent=False)
    table1 = doc.add_table(rows=1, cols=3)
    table1.style = "Table Grid"
    widths1 = [1900, 3000, 4460]
    headers1 = ["Function", "Hardware/software", "Role"]
    for idx, text in enumerate(headers1):
        fill_table_cell(table1.rows[0].cells[idx], text, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
        set_cell_shading(table1.rows[0].cells[idx], TABLE_HEADER_FILL)
    mark_table_header(table1.rows[0])
    rows1 = [
        ("Optical acquisition", "iPhone 15 Pro", "Multiview visible-light video acquisition"),
        ("Frame extraction", "FFmpeg [version]", "Extraction and selection of still frames"),
        ("3D reconstruction", "RealityCapture [version]", "SfM/MVS and GCP-constrained bundle adjustment"),
        ("3D appearance model", "PostShot [version]", "Generation and rendering of a 3D Gaussian representation"),
        ("DICOM processing", "Python; SimpleITK; NumPy; SciPy [versions]", "Physical-coordinate conversion, interpolation, and MPR generation"),
        ("Communication", "FastAPI; WebSocket [versions]", "Bidirectional transmission of view parameters and MPR images"),
        ("Rendering and user interface", "Unity [version]", "Optical model display, view control, and MPR overlay"),
    ]
    for row_data in rows1:
        cells = table1.add_row().cells
        for idx, value in enumerate(row_data):
            fill_table_cell(cells[idx], value, placeholder="[" in value)
    set_table_geometry(table1, widths1)

    add_plain(doc, "Table II. Technical accuracy results.", style="Caption", indent=False)
    table2 = doc.add_table(rows=1, cols=5)
    table2.style = "Table Grid"
    widths2 = [2200, 2200, 1000, 1500, 2460]
    headers2 = ["Endpoint", "Condition", "Models", "Observations", "Error, mean ± SD"]
    for idx, text in enumerate(headers2):
        fill_table_cell(
            table2.rows[0].cells[idx],
            text,
            bold=True,
            align=WD_ALIGN_PARAGRAPH.CENTER,
            size=9,
        )
        set_cell_shading(table2.rows[0].cells[idx], TABLE_HEADER_FILL)
    mark_table_header(table2.rows[0])
    rows2 = [
        ("Post-alignment geometric point error", "GCP-constrained", "3", "12", "0.27 ± 0.12 mm"),
        ("End-to-end TRE", "GCP-constrained", "3", "12", "0.45 ± 0.26 mm"),
        ("End-to-end TRE", "Unconstrained", "3", "12", "1.47 ± 0.95 mm"),
    ]
    for row_data in rows2:
        cells = table2.add_row().cells
        for idx, value in enumerate(row_data):
            align = WD_ALIGN_PARAGRAPH.LEFT if idx < 2 else WD_ALIGN_PARAGRAPH.CENTER
            fill_table_cell(cells[idx], value, align=align)
    set_table_geometry(table2, widths2)

    add_heading(doc, "Figure legends", 1)
    legends = [
        ("Figure 1.", "Workflow from CT and optical acquisition through CT-derived GCP localization, constrained optical reconstruction, DICOM-coordinate MPR generation, and integrated rendering."),
        ("Figure 2.", "Representative integrated display. The visible-light 3D model and view-driven CT MPR plane are rendered in the same physical coordinate system. Add panels showing the external view, MPR plane, and coordinate axes."),
        ("Figure 3.", "Geometric accuracy experiment. Show the phantom, the approximately 40-cm point distribution, correspondence definition, and the distinction between unaligned absolute error and post-alignment shape error."),
        ("Figure 4.", "End-to-end TRE experiment. Show the four non-GCP targets spanning approximately 30 cm and the measurement between the optical landmark and the corresponding MPR location."),
        ("Figure 5.", "Distribution of end-to-end TRE for GCP-constrained and unconstrained reconstruction. Plot all individual observations and indicate the model/run grouping; define the box, whiskers, and any confidence intervals in the final legend."),
    ]
    for label, legend in legends:
        p = doc.add_paragraph(style="Caption")
        r1 = p.add_run(f"{label} ")
        set_run_font(r1, size=10, bold=True)
        r2 = p.add_run(legend)
        set_run_font(r2, size=10, bold=False)

    add_heading(doc, "Working-draft completion checklist (remove before submission)", 1)
    checklist = [
        "Replace every yellow TO COMPLETE item with verified study information.",
        "Add unaligned absolute-coordinate accuracy and raw point-level data.",
        "Confirm the independence and pairing structure of observations and rerun the statistical analysis.",
        "Complete Figures 1–5 at publication resolution and with accessible colors.",
        "Verify all references and add missing literature on surface-to-CT registration and uncertainty.",
        "Remove this checklist and the DRAFT header after scientific and coauthor review.",
        "Create a clean blinded file with anonymized metadata, continuous line numbers, and page numbers.",
    ]
    for item in checklist:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.first_line_indent = Inches(0)
        p.paragraph_format.left_indent = Inches(0.5)
        p.paragraph_format.line_spacing = 1.5
        rr = p.add_run(item)
        set_run_font(rr)

    doc.save(MANUSCRIPT_PATH)


if __name__ == "__main__":
    add_title_page()
    add_main_manuscript()
    print(MANUSCRIPT_PATH)
    print(TITLE_PAGE_PATH)

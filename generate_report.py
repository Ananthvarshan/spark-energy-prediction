"""
generate_report.py
Generates a properly formatted Word document of the research paper.
Run: python generate_report.py
Output: outputs/Research_Report.docx
"""

import os
from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUTPUT_PATH = "outputs/Research_Report.docx"
os.makedirs("outputs", exist_ok=True)

doc = Document()

# ── Page margins ──────────────────────────────────────────────
for section in doc.sections:
    section.top_margin    = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin   = Cm(3.0)
    section.right_margin  = Cm(2.5)

# ── Helper: set paragraph font ────────────────────────────────
def set_run_font(run, size=11, bold=False, italic=False,
                 color=None, name="Calibri"):
    run.font.name  = name
    run.font.size  = Pt(size)
    run.font.bold  = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)

def add_paragraph(doc, text, style="Normal", size=11, bold=False,
                  italic=False, color=None, space_before=0,
                  space_after=6, align=WD_ALIGN_PARAGRAPH.LEFT):
    p = doc.add_paragraph(style=style)
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after  = Pt(space_after)
    p.paragraph_format.alignment    = align
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold, italic=italic, color=color)
    return p

def add_heading(doc, text, level=1):
    colors = {
        1: (31, 73, 125),   # deep blue
        2: (52, 90, 138),
        3: (68, 114, 196),
    }
    sizes  = {1: 16, 2: 13, 3: 12}
    space_before = {1: 18, 2: 14, 3: 10}

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before[level])
    p.paragraph_format.space_after  = Pt(4)
    if level == 1:
        # add bottom border
        pPr = p._p.get_or_add_pPr()
        pBdr = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "6")
        bottom.set(qn("w:space"), "1")
        bottom.set(qn("w:color"), "1E497B")
        pBdr.append(bottom)
        pPr.append(pBdr)

    run = p.add_run(text)
    set_run_font(run, size=sizes[level], bold=True,
                 color=colors[level], name="Calibri")
    return p

def add_bullet(doc, text, level=0, size=11):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.left_indent = Inches(0.25 + level * 0.25)
    run = p.add_run(text)
    set_run_font(run, size=size)
    return p

def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header row
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        for run in hdr_cells[i].paragraphs[0].runs:
            set_run_font(run, size=10, bold=True, color=(255, 255, 255))
        # Header background
        tc_pr = hdr_cells[i]._tc.get_or_add_tcPr()
        shd   = OxmlElement("w:shd")
        shd.set(qn("w:fill"), "1E497B")
        shd.set(qn("w:val"), "clear")
        tc_pr.append(shd)

    # Data rows
    for r_idx, row in enumerate(rows):
        cells = table.rows[r_idx + 1].cells
        bg    = "EEF3FB" if r_idx % 2 == 0 else "FFFFFF"
        for c_idx, val in enumerate(row):
            cells[c_idx].text = val
            for run in cells[c_idx].paragraphs[0].runs:
                set_run_font(run, size=10)
            tc_pr = cells[c_idx]._tc.get_or_add_tcPr()
            shd   = OxmlElement("w:shd")
            shd.set(qn("w:fill"), bg)
            shd.set(qn("w:val"), "clear")
            tc_pr.append(shd)

    # Column widths
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Inches(w)

    doc.add_paragraph()  # spacing after table
    return table

def add_code_block(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(4)
    p.paragraph_format.left_indent  = Inches(0.3)
    tc_pr = p._p.get_or_add_pPr()
    shd   = OxmlElement("w:shd")
    shd.set(qn("w:fill"), "F2F2F2")
    shd.set(qn("w:val"), "clear")
    tc_pr.append(shd)
    run = p.add_run(text)
    set_run_font(run, size=9, name="Courier New",
                 color=(50, 50, 50))
    return p

def add_note_box(doc, text, label="NOTE"):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after  = Pt(6)
    p.paragraph_format.left_indent  = Inches(0.2)
    run_label = p.add_run(f"{label}: ")
    set_run_font(run_label, size=10, bold=True, color=(31, 73, 125))
    run_text = p.add_run(text)
    set_run_font(run_text, size=10, italic=True, color=(60, 60, 60))
    return p

def hr(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(2)


# ════════════════════════════════════════════════════════════════
#  COVER PAGE
# ════════════════════════════════════════════════════════════════

add_paragraph(doc, "RESEARCH REPORT", size=9, color=(120, 120, 120),
              space_after=4, align=WD_ALIGN_PARAGRAPH.CENTER)

add_paragraph(doc, "Standby-Based Energy Optimization System\nfor Industrial Machines",
              size=22, bold=True, color=(31, 73, 125),
              space_before=30, space_after=10,
              align=WD_ALIGN_PARAGRAPH.CENTER)

add_paragraph(doc,
              "Using Gaussian Mixture Model (GMM) for State Detection\n"
              "and LSTM for Future State Prediction",
              size=13, italic=True, color=(68, 114, 196),
              space_after=30, align=WD_ALIGN_PARAGRAPH.CENTER)

add_paragraph(doc, "Repository: github.com/Ananthvarshan/spark-energy-prediction",
              size=10, italic=True, color=(100, 100, 100),
              align=WD_ALIGN_PARAGRAPH.CENTER)
add_paragraph(doc, "Branch: gmm-experiment   |   Last Updated: June 2026",
              size=10, italic=True, color=(100, 100, 100),
              align=WD_ALIGN_PARAGRAPH.CENTER, space_after=40)

doc.add_page_break()

# ════════════════════════════════════════════════════════════════
#  1. ABSTRACT
# ════════════════════════════════════════════════════════════════
add_heading(doc, "1. Abstract", 1)
add_paragraph(doc,
    "Industrial machines in manufacturing environments consume significant amounts of energy "
    "even when they are not actively performing work. This wasted energy — occurring during "
    "standby, idle, and off-peak periods — represents a major inefficiency in large-scale "
    "industrial operations. This research proposes a data-driven, unsupervised pipeline that "
    "automatically identifies the operational state of an industrial machine from its power "
    "consumption data, predicts future states using a sequence model, and generates "
    "energy-saving decisions accordingly.",
    space_after=6)
add_paragraph(doc,
    "The pipeline uses a Gaussian Mixture Model (GMM) for unsupervised state detection, a "
    "Viterbi Hidden Markov Model (HMM) for temporal smoothing of noisy transitions, and an "
    "LSTM (Long Short-Term Memory) neural network for future state prediction. The system "
    "operates entirely without human labelling of the data, making it broadly applicable "
    "across different machine types and industries.")

# ════════════════════════════════════════════════════════════════
#  2. PROBLEM STATEMENT
# ════════════════════════════════════════════════════════════════
add_heading(doc, "2. Problem Statement", 1)
add_paragraph(doc,
    "Industrial machines operate across four general power states: fully OFF, in STANDBY "
    "(powered on but inactive), actively WORKING, or at PEAK LOAD. A machine left in STANDBY "
    "unnecessarily wastes electrical energy continuously. This compounds over time — a single "
    "machine idling at 2 kW for 18 hours overnight adds up to hundreds of thousands of rupees "
    "in electricity costs per year at an industrial scale.")
add_paragraph(doc, "The core problem is:", bold=True, space_after=3)
add_note_box(doc,
    "Industrial machines spend a significant portion of their operating time consuming energy "
    "without producing any useful output. There is no automated, machine-agnostic system that "
    "can (a) identify these wasteful states from raw sensor data without manual labelling, and "
    "(b) predict when a machine is about to enter a long idle period so that it can be "
    "automatically shut down.", label="PROBLEM")
add_paragraph(doc,
    "Manual monitoring is impractical at scale. A factory with 22 machines cannot station an "
    "operator per machine to watch power consumption. The solution must be automated, "
    "unsupervised, and generalizable across different machine types.")

# ════════════════════════════════════════════════════════════════
#  3. OBJECTIVE
# ════════════════════════════════════════════════════════════════
add_heading(doc, "3. Objective", 1)
add_paragraph(doc, "The primary objectives of this research are:", space_after=4)
bullets = [
    "Automatic State Detection: Without any prior labelling, automatically classify every "
    "5-second power reading into one of three operational states: OFF, STANDBY, or WORKING.",
    "Noise-Robust Labelling: Remove physically impossible rapid transitions between states "
    "(sensor flicker) using a Viterbi HMM smoother.",
    "Future State Prediction: Using labelled historical data, train an LSTM to predict what "
    "state the machine will be in for the next N minutes (e.g., next 30 minutes).",
    "Decision Logic: Implement a rule-based layer that uses the LSTM's prediction to trigger "
    "energy-saving actions (e.g., 'If predicted STANDBY > 15 min → send shutdown command').",
    "Generalizability: The pipeline must work on any industrial machine dataset without "
    "requiring domain-specific manual configuration.",
]
for b in bullets:
    add_bullet(doc, b)

# ════════════════════════════════════════════════════════════════
#  4. RELATED WORK
# ════════════════════════════════════════════════════════════════
add_heading(doc, "4. Related Work & Background", 1)
add_paragraph(doc,
    "Industrial energy management is a well-studied field, but existing solutions fall into "
    "two categories:")
add_bullet(doc,
    "Rule-based systems: Use manually defined power thresholds per machine. Require domain "
    "expertise and fail when machines have varying baseline loads.")
add_bullet(doc,
    "Supervised learning systems: Require labelled datasets where a human expert has tagged "
    "each time period with a state. Labelled industrial datasets are rare and expensive.")
add_paragraph(doc, "What makes our approach different:", bold=True, space_before=8, space_after=3)
add_paragraph(doc,
    "Our approach is fully unsupervised. The GMM discovers states from the raw power "
    "distribution without any pre-defined thresholds. Additionally, unlike prior work that "
    "uses K-Means clustering, we use GMM, which models each state as a Gaussian distribution "
    "with its own mean and variance — more physically realistic than K-Means.")

# ════════════════════════════════════════════════════════════════
#  5. SYSTEM ARCHITECTURE
# ════════════════════════════════════════════════════════════════
add_heading(doc, "5. System Architecture", 1)
add_paragraph(doc, "The full pipeline consists of 10 steps:", space_after=6)
add_code_block(doc,
    "RAW DATA (timestamp, power)\n"
    "   ↓\n"
    "FEATURE ENGINEERING (hour_of_day, day_of_week, Δpower, rolling mean/std)\n"
    "   ↓\n"
    "GMM CLUSTERING (discovers OFF / STANDBY / WORKING clusters)\n"
    "   ↓\n"
    "HMM SMOOTHING (Viterbi — removes sensor-noise flicker)\n"
    "   ↓\n"
    "STATE MAPPING (low mean → OFF, medium → STANDBY, high → WORKING)\n"
    "   ↓\n"
    "LABELLED DATASET (timestamp | power | features | state)\n"
    "   ↓\n"
    "LSTM SEQUENCE MODEL (learns temporal patterns from labelled history)\n"
    "   ↓\n"
    "FUTURE STATE PREDICTION (predicts next N time steps)\n"
    "   ↓\n"
    "DECISION LOGIC (if predicted STANDBY > threshold → trigger action)\n"
    "   ↓\n"
    "ENERGY SAVING ACTION (manual suggestion OR automatic control signal)")
add_paragraph(doc,
    "Model X (State Detection): GMM → HMM Smoother → State Labels",
    bold=True, space_before=6, space_after=3)
add_paragraph(doc,
    "Model Y (State Prediction): LSTM → Future State Sequence → Decision Engine",
    bold=True, space_after=6)

# ════════════════════════════════════════════════════════════════
#  6. METHODOLOGY
# ════════════════════════════════════════════════════════════════
add_heading(doc, "6. Methodology", 1)

add_heading(doc, "6.1  Data Loading and Preparation", 2)
bullets = [
    "Source: SPARK industrial energy dataset (.csv.xz compressed files)",
    "Each file contains timestamped power readings at 5-second intervals",
    "Data cleaning: NaN values dropped, negative power clipped to zero, timestamp parsed",
    "Power column auto-detected regardless of original column name (P_total, I1, I2, I3, etc.)",
]
for b in bullets:
    add_bullet(doc, b)

add_heading(doc, "6.2  Gaussian Mixture Model (GMM) — State Detection", 2)
add_paragraph(doc,
    "GMM is an unsupervised probabilistic model that assumes data is generated from a mixture "
    "of K Gaussian distributions. Each Gaussian represents one machine state.")

add_paragraph(doc, "Why GMM instead of K-Means?", bold=True, space_before=6, space_after=4)
add_table(doc,
    ["Property", "K-Means", "GMM"],
    [
        ["Cluster shape", "Circular, equal size", "Elliptical, variable size"],
        ["Assignment", "Hard (one cluster only)", "Soft (probability per cluster)"],
        ["Physical realism", "Lower", "Higher"],
        ["Handles overlapping states", "No", "Yes"],
    ],
    col_widths=[2.0, 2.2, 2.2])

add_paragraph(doc, "GMM parameters used:", bold=True, space_after=4)
bullets = [
    "k = 3 (fixed after validation; represents OFF, STANDBY, WORKING)",
    "n_init = 3 (three random restarts to avoid local optima)",
    "covariance_type = 'full'",
    "random_state = 42 (for reproducibility)",
]
for b in bullets:
    add_bullet(doc, b)

add_heading(doc, "6.3  Viterbi HMM Smoothing", 2)
add_paragraph(doc,
    "Even after GMM labelling, the raw label sequence contains physically impossible "
    "transitions — for example, WORKING → OFF → WORKING within 15 seconds. This is sensor "
    "noise. The Viterbi algorithm finds the single most probable state sequence over the "
    "entire timeline, penalizing rapid state changes. This removes spurious flicker while "
    "preserving real transitions.")
add_note_box(doc,
    "Key parameter: stay_probability = 0.995 — the machine is expected to stay in its "
    "current state 99.5% of the time at each 5-second step. Any label change costs "
    "probability — only real transitions survive.")

add_heading(doc, "6.4  LSTM Future State Prediction (Planned)", 2)
add_paragraph(doc,
    "An LSTM neural network will be trained on HMM-smoothed label sequences to learn "
    "temporal patterns such as: machine is always OFF on Sundays after 6 PM; STANDBY episodes "
    "last 2–20 minutes on average; and day-of-week specific working schedules.")

# ════════════════════════════════════════════════════════════════
#  7. DATASET
# ════════════════════════════════════════════════════════════════
add_heading(doc, "7. Dataset", 1)
add_table(doc,
    ["Property", "Value"],
    [
        ["Dataset name", "SPARK (Smart inPustrial cAmpus Research networK)"],
        ["Number of machines", "22 industrial machines + 1 solar PV system"],
        ["Machine types", "CNC mills, lathes, chip presses, pumps, pick-and-place robots"],
        ["Sampling rate", "Every 5 seconds"],
        ["Data span", "1 to 7 years per machine"],
        ["Format", ".csv.xz (compressed CSV)"],
        ["Columns used", "WsDateTime (timestamp), P_total (power in Watts)"],
    ],
    col_widths=[2.5, 4.0])

add_paragraph(doc, "Files tested so far:", bold=True, space_after=4)
add_bullet(doc, "2024_P_total.csv.xz — CNC Machine, full year 2024, 6,017,356 rows")
add_bullet(doc, "2024_AC_ActivePower.csv.xz — Solar Panel inverter, full year 2024")
add_bullet(doc, "2024_P_total_PickAndPlace.csv.xz — EPI Pick-and-Place Robot")

# ════════════════════════════════════════════════════════════════
#  8. IMPLEMENTATION
# ════════════════════════════════════════════════════════════════
add_heading(doc, "8. Implementation", 1)
add_paragraph(doc, "Language: Python 3.13", bold=True, space_after=4)
add_table(doc,
    ["Library", "Purpose"],
    [
        ["pandas", "Data loading, cleaning, time-series manipulation"],
        ["numpy", "Numerical operations, array math"],
        ["scikit-learn", "GMM (GaussianMixture), Silhouette score"],
        ["scipy", "Gaussian filter, histogram peak detection"],
        ["matplotlib", "All diagnostic and output plots"],
    ],
    col_widths=[2.0, 4.5])

add_paragraph(doc, "File structure:", bold=True, space_before=8, space_after=4)
add_code_block(doc,
    "project/\n"
    "├── main.py              # Entry point: single or multi-machine analysis\n"
    "├── validate_gmm.py      # Standalone 6-test GMM validation harness\n"
    "├── src/\n"
    "│   └── data_analysis.py # Core pipeline: data loader, GMM, HMM, plots\n"
    "├── data/\n"
    "│   └── *.csv.xz         # Machine datasets\n"
    "└── outputs/\n"
    "    ├── gmm_validation/  # Validation harness output plots\n"
    "    └── plots_*/         # Per-machine analysis output plots")

# ════════════════════════════════════════════════════════════════
#  9. VALIDATION FRAMEWORK
# ════════════════════════════════════════════════════════════════
add_heading(doc, "9. Validation Framework", 1)
add_paragraph(doc,
    "A key contribution of this research is a 6-test independent validation harness "
    "(validate_gmm.py) built to prove that the GMM is splitting data correctly before any "
    "downstream models are built on top of it.")
add_table(doc,
    ["Test", "What It Checks", "Pass Criterion"],
    [
        ["T1 — Data Sanity", "No NaN, no negatives, sufficient rows", "All basic quality checks pass"],
        ["T2 — GMM Convergence", "EM algorithm actually converged", "gmm.converged_ == True for all k"],
        ["T3 — k Selection", "Silhouette and BIC agree on same k", "Both metrics select same k"],
        ["T4 — Physical Separation (2σ)", "Adjacent cluster means > 2 std devs apart", "Ratio >= 2.0 for all adjacent pairs"],
        ["T5 — Soft-Assignment Confidence", "Model assigns readings with high probability", ">70% of readings have >80% confidence"],
        ["T6 — Weight Sanity", "No cluster has < 0.5% of the data", "All cluster weights >= 0.5%"],
    ],
    col_widths=[2.2, 2.5, 2.0])

# ════════════════════════════════════════════════════════════════
#  10. RESULTS
# ════════════════════════════════════════════════════════════════
add_heading(doc, "10. Results", 1)
add_heading(doc, "10.1  CNC Machine — Full Year 2024", 2)
add_table(doc,
    ["Property", "Value"],
    [
        ["Total data rows", "6,017,356 (one full year at 5-second intervals)"],
        ["Power range", "0 W → 59,576 W"],
        ["Mean power (all time)", "687.69 W"],
        ["Standard deviation", "1,664.47 W"],
    ],
    col_widths=[3.0, 3.5])

add_paragraph(doc, "GMM discovered states (k = 3):", bold=True, space_before=10, space_after=4)
add_table(doc,
    ["State", "Mean Power", "Std Dev", "% of Year", "Total Hours"],
    [
        ["OFF",      "~0 W",     "~0 W",     "77.0%",  "6,757 h"],
        ["STANDBY",  "~2,177 W", "~1,000 W", "18.87%", "1,655 h"],
        ["WORKING",  "~6,549 W", "~521 W",   "4.06%",  "356 h"],
    ],
    col_widths=[1.4, 1.5, 1.2, 1.3, 1.3])

add_note_box(doc,
    "Key finding: This CNC machine was actively cutting/working for only 4% of the entire "
    "year. It spent 77% of the year completely powered off and nearly 19% powered on but "
    "doing nothing useful (STANDBY) — a massive energy waste opportunity.")

add_heading(doc, "10.2  Validation Test Results", 2)
add_table(doc,
    ["Test", "Result", "Notes"],
    [
        ["T1 Data Sanity",           "PASS", "Clean data, no issues"],
        ["T2 GMM Convergence",       "PASS", "Converged in 5-6 EM iterations"],
        ["T3 k Selection",           "WARN", "Silhouette=k3, BIC=k4; k=3 confirmed correct"],
        ["T4 Physical Separation",   "PASS", "OFF→STANDBY: 4.35σ; STANDBY→WORKING: 5.75σ"],
        ["T5 Confidence",            "PASS", "99.9% of readings assigned with >80% confidence"],
        ["T6 Weight Sanity",         "PASS", "All 3 final states have meaningful weights"],
    ],
    col_widths=[2.2, 1.2, 3.0])

# ════════════════════════════════════════════════════════════════
#  11. KNOWN LIMITATIONS
# ════════════════════════════════════════════════════════════════
add_heading(doc, "11. Known Limitations & Conceptual Issues", 1)

add_heading(doc, "11.1  Power-Only Clustering Cannot Distinguish STANDBY from Light-Load Working", 2)
add_note_box(doc,
    "This is the most important conceptual limitation of the current approach.",
    label="IMPORTANT")
add_paragraph(doc,
    "GMM clusters data based only on power magnitude. It has no concept of whether the "
    "machine is 'doing work' or 'idling.' As a result, a machine consuming 2,000 W while "
    "idling (true STANDBY) and a machine consuming 2,200 W while running a very light job "
    "(active at low capacity) will both be assigned to the STANDBY cluster.")
add_paragraph(doc,
    "What this means practically: The STANDBY label should be interpreted as a medium-power "
    "consumption tier, not a guarantee that the machine is doing zero productive work. "
    "For energy analysis, both true standby and light-load work represent inefficiency "
    "relative to peak productive output, so this ambiguity has limited practical impact.")
add_paragraph(doc,
    "Potential fix: Adding multi-dimensional features (phase-specific currents, vibration "
    "sensors, PLC signals) would allow GMM to distinguish these cases. However, our current "
    "dataset only provides a single P_total column.")

add_heading(doc, "11.2  k-Selection Disagreement Between Silhouette and BIC", 2)
add_paragraph(doc,
    "During validation, Silhouette score (geometric) chose k=3 while BIC (probabilistic) "
    "chose k=4. The 4th cluster (PEAK_LOAD) had a weight of only 0.05% — a ghost cluster "
    "capturing extreme sensor noise, not a real physical state. Fix implemented: k was "
    "hardcoded to 3 in the production pipeline after validation confirmed it as correct.")

add_heading(doc, "11.3  Large Dataset Processing Time", 2)
add_paragraph(doc,
    "The Viterbi HMM algorithm runs a pure-Python loop over all T time steps. For 6 million "
    "rows, this takes approximately 60 seconds. For machines with 5+ years of data "
    "(~30 million rows), this would take several minutes.")

# ════════════════════════════════════════════════════════════════
#  12. DIFFICULTIES FACED
# ════════════════════════════════════════════════════════════════
add_heading(doc, "12. Difficulties Faced", 1)

add_heading(doc, "12.1  Technical Difficulties", 2)
difficulties = [
    ("Windows terminal encoding (cp1252)",
     "Python's default encoding on Windows could not render Unicode box-drawing characters "
     "used in the validation output. Fixed by reconfiguring stdout to UTF-8 at script startup."),
    ("Git branch switching with uncommitted changes",
     "When switching from the main branch to gmm-experiment, Git blocked the checkout "
     "because main.py had uncommitted local modifications. Solution: git stash to temporarily "
     "save changes, then switch branches."),
    ("GMM ghost cluster (k=4)",
     "The BIC score recommended k=4 even though the 4th cluster represented only 0.05% of "
     "the data. This required building the T6 Weight Sanity test in the validation harness "
     "to programmatically catch this case."),
    ("Silhouette score bias toward fewer clusters",
     "Silhouette score has a known bias toward fewer, more widely-separated clusters even "
     "when more clusters are statistically justified. This is why both Silhouette AND BIC "
     "are used as independent checks."),
    ("Power column detection across different SPARK files",
     "Different machine files in SPARK use different column names (P_total, I1, I2, I3, "
     "power.i1, etc.). The data loader required a flexible multi-candidate detection strategy."),
]
for i, (title, desc) in enumerate(difficulties, 1):
    add_paragraph(doc, f"{i}. {title}", bold=True, space_before=6, space_after=2)
    add_paragraph(doc, desc, space_after=4)

add_heading(doc, "12.2  Conceptual Difficulties", 2)
conceptual = [
    ("Defining STANDBY precisely",
     "GMM cannot distinguish a machine in true idle mode from one doing light work at medium "
     "power. This is a fundamental limitation of single-dimensional power-based clustering."),
    ("Choosing the right number of clusters (k)",
     "The choice of k=3 vs k=4 required running three independent validation methods "
     "(Silhouette, BIC, histogram peak counting) and manually inspecting the output. "
     "There is no single definitive automatic answer."),
]
for i, (title, desc) in enumerate(conceptual, 1):
    add_paragraph(doc, f"{i}. {title}", bold=True, space_before=6, space_after=2)
    add_paragraph(doc, desc, space_after=4)

# ════════════════════════════════════════════════════════════════
#  13. HARDWARE
# ════════════════════════════════════════════════════════════════
add_heading(doc, "13. Hardware Used", 1)
add_table(doc,
    ["Component", "Specification"],
    [
        ["Operating System",  "Windows 11"],
        ["Python version",    "Python 3.13"],
        ["Storage",           "Dataset files: ~14 MB compressed (SPARK 2024 subset)"],
        ["RAM",               "Standard laptop RAM; 6M-row dataset loaded via pandas"],
        ["GPU",               "Not used at this stage (GMM and HMM are CPU-bound)"],
        ["Cloud / HPC",       "Not used; all processing done locally"],
    ],
    col_widths=[2.5, 4.0])
add_note_box(doc,
    "For the LSTM training stage (next step), a GPU or cloud resource (e.g., Google Colab) "
    "may be required for reasonable training times.")

# ════════════════════════════════════════════════════════════════
#  14. NEXT STEPS
# ════════════════════════════════════════════════════════════════
add_heading(doc, "14. Next Steps / Future Work", 1)

add_heading(doc, "Immediate (Next 2–4 Weeks)", 2)
bullets = [
    "Fix k-selection logic in data_analysis.py with a ghost cluster safety check: if any "
    "cluster weight < 1%, fall back to Silhouette selection.",
    "Run validation on all available SPARK datasets (solar panel, pick-and-place robot) "
    "using validate_gmm.py to compare GMM performance across machine types.",
    "Complete labelled dataset generation — run full pipeline (GMM + HMM) on all machines "
    "to produce the labelled CSV needed to train LSTM.",
]
for b in bullets:
    add_bullet(doc, b)

add_heading(doc, "Medium Term (Next 1–2 Months)", 2)
bullets = [
    "Build the LSTM sequence model (Model Y) — train on labelled sequences to predict "
    "future machine states.",
    "Implement the decision logic layer — define the rule: if predicted STANDBY duration "
    "> threshold → suggest OFF.",
    "Add multi-dimensional features — if data with phase-specific currents or other sensor "
    "columns is available, extend GMM to resolve the STANDBY vs. light-load ambiguity.",
]
for b in bullets:
    add_bullet(doc, b)

add_heading(doc, "Long Term", 2)
bullets = [
    "Test on a factory dataset with strict scheduled working hours — the system is most "
    "useful when machines have predictable patterns.",
    "Build an API or dashboard — expose the prediction as a REST API so factory management "
    "systems can query it in real time.",
]
for b in bullets:
    add_bullet(doc, b)

# ════════════════════════════════════════════════════════════════
#  15. OUR HONEST OPINION
# ════════════════════════════════════════════════════════════════
add_heading(doc, "15. Our Honest Opinion on This Research", 1)
add_note_box(doc,
    "This section reflects our candid assessment after approximately two months of deep "
    "work on this topic. We believe transparency is important when doing research.",
    label="NOTE")

add_heading(doc, "What We Have Built", 2)
add_paragraph(doc,
    "We have built a working, validated pipeline. The GMM correctly identifies machine "
    "states from raw power data. The Viterbi HMM smoother removes sensor noise. The "
    "validation harness provides mathematical proof that the clustering is physically "
    "meaningful. This is real, functional research-grade code.")

add_heading(doc, "Our Concern About the Topic", 2)
add_paragraph(doc,
    "However, as we have gone deeper into this field, we have started to question whether "
    "the problem is as unique and impactful as we initially believed.")

add_paragraph(doc, "The problem is not unsolved.", bold=True, space_before=6, space_after=3)
add_paragraph(doc,
    "Energy monitoring and state detection for industrial machines already exists as a "
    "commercial product category (e.g., Siemens SIMATIC Energy Manager, Eaton Power Xpert). "
    "The novelty of our approach — using GMM + LSTM instead of manual rule configuration — "
    "is real, but the problem space is more 'an improvement on existing solutions' rather "
    "than a breakthrough discovery.")

add_paragraph(doc, "Our solution is highly context-specific.", bold=True, space_before=6, space_after=3)
add_paragraph(doc, "The system works best when applied to:", space_after=4)
add_bullet(doc, "Large companies or factories that are mostly autonomous")
add_bullet(doc, "Factories with strictly scheduled working hours (e.g., two fixed 8-hour shifts)")
add_bullet(doc, "Factories that operate continuously, 24/7")
add_paragraph(doc,
    "In these environments, the LSTM can learn very precise patterns: 'every weeknight at "
    "10 PM, the line shuts down for 6 hours.' The energy savings would be large and "
    "predictable.")

add_paragraph(doc, "The dataset problem.", bold=True, space_before=6, space_after=3)
add_paragraph(doc,
    "Factories that match the ideal profile above — highly autonomous, strictly scheduled, "
    "large-scale — are exactly the factories least likely to share their operational data "
    "publicly. We currently have the SPARK dataset, which is the best publicly available "
    "industrial energy dataset. However, SPARK was collected from a research campus, not "
    "an autonomous mass-production factory. The machine schedules in SPARK are somewhat "
    "irregular, which limits how well the LSTM can learn clean temporal patterns.")

add_note_box(doc,
    "We are not fully confident that our current dataset will produce strong LSTM prediction "
    "results. The STANDBY episodes in SPARK may not follow a consistent enough schedule for "
    "the LSTM to reliably predict long standby periods in advance.",
    label="CONCERN")

add_heading(doc, "The Hard Question: Should We Change the Topic?", 2)
add_paragraph(doc,
    "We have worked on this for approximately two months. Here is our honest view:")

add_paragraph(doc, "Reasons to continue:", bold=True, space_before=6, space_after=3)
add_bullet(doc, "The core GMM + HMM pipeline is working correctly and is validated")
add_bullet(doc, "The topic is technically sound and the code is clean and well-structured")
add_bullet(doc,
    "There is genuine value in the unsupervised approach — no other public work combines "
    "GMM with Viterbi HMM specifically for industrial machine state detection")
add_bullet(doc, "Two months of deep work has given us real domain expertise in this area")

add_paragraph(doc, "Reasons to reconsider:", bold=True, space_before=6, space_after=3)
add_bullet(doc,
    "The ideal dataset (strictly-scheduled autonomous factory) may be impossible to obtain")
add_bullet(doc,
    "The LSTM prediction quality is unknown until we build and test it")
add_bullet(doc,
    "If LSTM predictions are poor due to irregular schedules in SPARK, the entire "
    "decision logic layer becomes unreliable")

add_paragraph(doc, "Our recommendation:", bold=True, space_before=8, space_after=3)
add_paragraph(doc,
    "Before making a final decision, we should complete the LSTM training on the existing "
    "SPARK labelled data and measure its prediction accuracy. If the accuracy is reasonable "
    "(>75% on a held-out test set), the topic is worth finishing. If the LSTM cannot learn "
    "meaningful patterns from SPARK's irregular schedules, we should either (a) seek a "
    "better dataset or (b) scope the paper to only cover Model X (GMM + HMM state detection) "
    "and position the LSTM as 'future work.'")

# ════════════════════════════════════════════════════════════════
#  SAVE
# ════════════════════════════════════════════════════════════════
doc.save(OUTPUT_PATH)
print(f"\nWord document saved to: {OUTPUT_PATH}")
print("Open it in Microsoft Word or LibreOffice Writer.")

#!/usr/bin/env python3
# =============================================================================

# COLOURS OF RECORD.
#   REPLACED 2026-09-30: hue = controller family, shade = role in its ablation pair.
#   DARK = the arm of record in each pair (S1O policy of record, A2 alternative explored, PID-baseline comparator);
#   LIGHT = its offset-ablation twin (S1, A2O, PIDO). S family blue, A family brown, PID family grey (operator chose grey
#   over pink: neutral "reference" colour, prints in greyscale, no clash with the red/orange/green rule lines).
#   Markers follow the same rule: arm of record filled + solid line; twin hollow + dotted line. Old palette and the
#   figures drawn with it: archive_palette_20260930/.
#   Guard trip / H1 bar 2.00 m = dashed orange. Airframe contact plane 3.42 m = solid red.
# RULES: no title inside a figure and no figure number anywhere; files are named
#   for their content; arm order and arm names are IMPORTED from the verdict tool, never re-typed.
# CLI: (none; imported) SIDE-EFFECTS: save() writes <name>.png and <name>.pdf under OUT/figures only.
# =============================================================================
import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, (_REPO + "/scoring/tools"))
from offset_delivery import OUT, RUN, md  # noqa: E402,F401
import hypothesis_verdicts as V  # noqa: E402

ARMS = list(V.ORDER)                                   # A2, A2O, S1, S1O, PID, PIDO
# 2026-10-01 v3: S family blue, A family purple, PID family brown; DARK = the arm WITH the offset (S1O, A2O, PIDO),
# LIGHT = the arm WITHOUT it (S1, A2, PID). Filled marker = with the offset, hollow = without (so colour and marker say the same thing).
COLOUR = {"S1O": "#1f77b4", "S1": "#9ecae1", "A2O": "#7b3294", "A2": "#c2a5cf", "PIDO": "#8c564b", "PID": "#d9b8a9"}
DARK = {"S1O": "#1f77b4", "S1": "#1f77b4", "A2O": "#7b3294", "A2": "#7b3294", "PIDO": "#8c564b", "PID": "#8c564b"}   # edge / mark colour: the family's dark shade
PALETTE_RULE = "dark shade and hollow marker = the arm with the 0.25 m offset (S1O, A2O, PIDO); light shade and filled marker = the same controller without it (S1, A2, PID)"
ROW_ORDER = ["S1O", "S1", "A2O", "A2", "PIDO", "PID"]                    # S family first, PID-baseline at the bottom; the order does not depend on which arm is the policy of record
LABEL = {"A2": "FlightDock-A2", "A2O": "FlightDock-A2O", "S1": "FlightDock-S1", "S1O": "FlightDock-S1O",
         "PID": "PID-baseline (comparator)", "PIDO": "PIDO"}
SHORT = {"A2": "A2", "A2O": "A2O", "S1": "S1", "S1O": "S1O", "PID": "PID", "PIDO": "PIDO"}   # 2026-10-01: abbreviated on figures to save space
MARKER = {"A2": "o", "A2O": "o", "S1": "s", "S1O": "s", "PID": "^", "PIDO": "^"}             # line charts: arm of record filled + solid line,
LINESTYLE = {"S1O": "-", "A2O": "-", "PIDO": "-", "S1": ":", "A2": ":", "PID": ":"}       # 2026-10-01 operator: solid = WITH the offset (hollow marker), dotted = without (filled marker)
HOLLOW = {"S1O", "A2O", "PIDO"}   # 2026-10-01 v4: marker fill SWAPPED, hollow = WITH the offset, filled = without; shades unchanged
# 2026-10-01: the twin's light tint must be VISIBLE, so a twin's
# whisker / line is drawn in its light COLOUR with a hollow marker edged in the family's DARK; the arm of record is drawn in
# DARK with a filled marker. Lines thicker so the colour reads. No description text inside a figure (the operator adds it in
# the document from the .md notes, which also carry a suggested title). Legend at the RIGHT side, never a thin bar below.
# the document figures are wider than tall. Rule lines: bars dotted.
LW_WHISKER, LW_LINE, MS = 3.2, 2.4, 7   # 2026-10-01: lighter at print size
PCT = "%"   # 2026-10-01 operator: no space before the percent sign anywhere ("95%", not "95 %")


def whisker_colour(k):
    """Dark for the arm of record, the light tint for its twin (so the tint is visible even with a hollow marker)."""
    return COLOUR[k]


def marker_face(k):
    return "white" if k in HOLLOW else COLOUR[k]
GUARD_TRIP, CONTACT = V.BAR, V.CONTACT                 # 2.00 m and 3.42 m, read from the verdict tool
ORANGE, RED = "#ff7f0e", "#d62728"
FIGDIR = os.path.join(OUT, "figures")
SEED_BLOCK = list(range(10775, 10805))
# 2026-10-01 PRINT SIZE: figures are built at the width they print at, so Word never scales them. US Letter;
#   portrait margins 1.25 in left and right -> text width 6.0 in; landscape margins 1 in left and right -> 9.0 in wide,
#   text height 8.5 - 1.25 - 1.25 - 0.5 header - 0.5 footer = 5.0 in. Figure font 8 pt.
TEXT_W_PORTRAIT = 8.5 - 1.25 - 1.25
TEXT_W_LANDSCAPE = 11.0 - 1.0 - 1.0
TEXT_H_LANDSCAPE = 8.5 - 1.25 - 1.25 - 0.5 - 0.5
# 2026-10-01 operator ("Font is too big! ... Don't be beholden to 8 pt"): base 7 pt, row text 6.5 pt, the panel title on ONE line
#   sized by fit_title() to span the axes width; the legend is OMITTED from chapter figures (the operator adds a 9 pt note under
#   the figure in Word from the .md "Figure note"). Whiskers and markers lighter.
FONT, FONT_ROW, FONT_TITLE_MAX = 7, 5.0, 8   # 2026-10-01: row text 5.0 pt
# 2026-10-01 SLIDE MODE: FIG_SLIDE=1 in the environment renders the SAME scripts, same
#   numbers and validation, at 10 in wide with slide-sized text, no crop note, into FIGDIR/slides_20261001/<STEM>_SLIDE.png|.pdf.
SLIDE = os.environ.get("FIG_SLIDE", "") == "1"
SLIDE_W = 10.0
if SLIDE:
    FONT, FONT_ROW, FONT_TITLE_MAX = 12, 9.0, 14
    LW_WHISKER, LW_LINE, MS = 5.0, 3.6, 11
    # scale every figure to SLIDE_W wide AT CREATION (the title fit and the text-edge fit measure the live figure), so the
    # scripts' print-size figsize calls are wrapped: width -> SLIDE_W, height in proportion.
    _subplots, _figure = plt.subplots, plt.figure
    def _scale(kw):
        if "figsize" in kw and kw["figsize"] is not None:
            w, h = kw["figsize"]
            kw["figsize"] = (SLIDE_W, h * SLIDE_W / w)
        return kw
    def _sub(*a, **kw): return _subplots(*a, **_scale(kw))
    def _fig(*a, **kw): return _figure(*a, **_scale(kw))
    plt.subplots, plt.figure = _sub, _fig
GREEN = "#2ca02c"
DPI = 300
plt.rcParams.update({"font.size": FONT, "axes.labelsize": FONT, "axes.titlesize": FONT, "xtick.labelsize": FONT, "ytick.labelsize": FONT,
                     "legend.fontsize": FONT_ROW + 0.5 if SLIDE else FONT, "figure.dpi": 100})


def fit_title(fig, ax, text, align="left"):
    """One-line title whose width equals the axes width: font size = FONT_TITLE_MAX scaled by (axes width / measured width),
    never above FONT_TITLE_MAX, never below 5.5 pt (refuses). Measured, not typed."""
    t = ax.set_title(text, loc=align, fontsize=FONT_TITLE_MAX)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    size = min(FONT_TITLE_MAX, FONT_TITLE_MAX * ax.get_window_extent(r).width / t.get_window_extent(r).width)
    if size < (9.0 if SLIDE else 5.5):
        raise SystemExit("REFUSE: title needs %.1f pt to fit on one line" % size)
    t.set_fontsize(size)
    print("title fitted at %.1f pt" % size)
    return t


def fit_text_edge(fig, ax, texts, x_edge, x_text, pad=0.015):
    """Set the x limit on the TEXT side so the widest of `texts` (ha="left" at x_text) just fits. x_edge is the limit on the
    other side (already set). With text width w px and axis width W px, f = w / W; the text starts at display fraction
    (x_text - x_edge) / (x_far - x_edge); solving fraction + f + pad = 1 gives x_far. Works on a normal and a reversed axis.
    Refuses if any text still ends outside the axis. Measured, not typed."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    f = max(t.get_window_extent(r).width for t in texts) / ax.get_window_extent(r).width
    x_far = x_edge + (x_text - x_edge) / (1.0 - f - pad)
    ax.set_xlim(x_edge, x_far)
    fig.canvas.draw()
    for t in texts:
        bb, ab = t.get_window_extent(r), ax.get_window_extent(r)
        if bb.x1 > ab.x1 + 0.5:
            raise SystemExit("REFUSE: row text overruns the axis (%.1f px > %.1f px)" % (bb.x1, ab.x1))
    tb, ab = ax.title.get_window_extent(r), ax.get_window_extent(r)
    if tb.x1 > ab.x1 + 0.5:
        raise SystemExit("REFUSE: panel title overruns its axis (%.1f px > %.1f px)" % (tb.x1, ab.x1))
    print("x range fitted: %.2f to %.2f (text needs %.1f%% of the axis width)" % (x_edge, x_far, 100 * f))
    return x_far


def place_legend(fig, ax, handles, avoid, loc="lower left"):
    fig.set_layout_engine("constrained")
    """Legend INSIDE the axes at `loc` if it overlaps none of the `avoid` artists; otherwise below the axes in two columns
    (2026-10-01: on a print-width chart a legend at the right would cost a quarter of the width)."""
    leg = ax.legend(handles=handles, loc=loc, framealpha=0.95, borderaxespad=0.6)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    lb = leg.get_window_extent(r)
    hit = [a for a in avoid if a.get_window_extent(r).overlaps(lb)]
    if hit:
        leg.remove()
        leg = fig.legend(handles=handles, loc="outside lower center", ncol=2, framealpha=0.95)   # constrained layout reserves the room
        print("legend placed BELOW the axes (inside placement overlapped %d artists)" % len(hit))
    else:
        print("legend placed inside the axes at", loc)
    return leg


def crop_note(fig, ax, text, width_in):
    if SLIDE:   # no note on a slide: drop the note axes and remember its share of the height so save_exact can trim it
        fig.canvas.draw()
        fig._slide_trim = ax.get_position().height
        ax.remove()
        print("slide mode: crop note omitted")
        return 0
    """The figure note, as printed under the figure in the document. Drawn in its own axes (axis off), wrapped to the figure width by
    measuring a sample line, separated by a dashed rule."""
    import textwrap
    ax.set_axis_off()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    probe = ax.text(0, 0, "x" * 100, fontsize=FONT_ROW)
    per_char_in = probe.get_window_extent(r).width / 100 / fig.dpi
    probe.remove()
    ax.axhline(1.0, xmin=0, xmax=1, color="0.5", lw=0.8, ls=(0, (3, 3)), clip_on=False)
    fs = FONT_ROW
    while True:   # the note never moves the figure above the rule: it stays out of the layout and shrinks until it fits its own axes
        cols = int((width_in - 0.2) / (per_char_in * fs / FONT_ROW)); lines = textwrap.wrap(text, cols)
        t = ax.text(0.0, 0.92, "\n".join(lines), ha="left", va="top", fontsize=fs, linespacing=1.3, transform=ax.transAxes); t.set_in_layout(False)
        fig.canvas.draw()
        if t.get_window_extent(r).height <= ax.get_window_extent(r).height * 0.9 or fs <= 4.0: break
        t.remove(); fs -= 0.25
    print("crop note: %d lines at %.1f pt" % (len(lines), fs))
    return len(lines)


def save_exact(fig, name, width_in):
    """Save at exactly width_in inches wide (no tight bbox), constrained layout, DPI; verify the PNG pixel width.
    Slide mode: the figure is scaled to SLIDE_W wide (height in proportion, minus the dropped note) and saved as <name>_SLIDE."""
    from PIL import Image
    outdir = FIGDIR
    if SLIDE:   # the figure is already SLIDE_W wide (creation-time scaling); only the dropped note's height is trimmed
        w0, h0 = fig.get_size_inches()
        width_in, name, outdir = SLIDE_W, name + "_SLIDE", os.path.join(FIGDIR, "slides_20261001")
        fig.set_size_inches(width_in, h0 * (1.0 - getattr(fig, "_slide_trim", 0.0)))
    fig.set_size_inches(width_in, fig.get_size_inches()[1])
    fig.set_layout_engine("constrained")
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        if SLIDE:   # slides: trim the margins (the dropped note leaves a blank band under a constrained layout)
            fig.savefig(os.path.join(outdir, name + "." + ext), dpi=DPI, bbox_inches="tight", pad_inches=0.08)
        else:
            fig.savefig(os.path.join(outdir, name + "." + ext), dpi=DPI)
    plt.close(fig)
    FIG_PATH = os.path.join(outdir, name)
    w, h = Image.open(FIG_PATH + ".png").size
    if not SLIDE and w != round(width_in * DPI):
        raise SystemExit("REFUSE: PNG width %d px is not %.2f in at %d dpi" % (w, width_in, DPI))
    print("wrote %s.png|.pdf at %.2f x %.2f in (%d x %d px, %d dpi)" % (FIG_PATH, w / DPI, h / DPI, w, h, DPI))


def reference_lines(ax, x_text):
    ax.axhline(GUARD_TRIP, color=ORANGE, ls="--", lw=1.4, zorder=1)
    ax.axhline(CONTACT, color=RED, ls="-", lw=1.8, zorder=1)
    ax.text(x_text, GUARD_TRIP + 0.04, "guard trip / H1 bar, > %.2f m" % GUARD_TRIP, color=ORANGE, fontsize=8, va="bottom", ha="right")
    ax.text(x_text, CONTACT + 0.04, "airframe contact plane, %.2f m" % CONTACT, color=RED, fontsize=8, va="bottom", ha="right")


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, name + "." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(FIGDIR, name + ".png|.pdf"))

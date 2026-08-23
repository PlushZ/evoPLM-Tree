#!/usr/bin/env python3
"""
Shared PLOS figure-submission compliance helpers: font resolution, dpi/format
constants, multi-format export (png/pdf/tiff/eps), Ghostscript EPS re-compression,
and side-by-side panel composition (used to build the final lettered multi-panel
figures from individually generated panels).
"""

import shutil
import subprocess
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

DPI = 300  # PLOS: 300-600 dpi
FORMATS = ("png", "pdf", "tiff", "eps")
FONT_FALLBACKS = ["Arial", "Liberation Sans", "DejaVu Sans"]


def apply_rcparams() -> None:
    """(Re-)assert PLOS-compliant rcParams; safe to call again after plt.style.use()."""
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = FONT_FALLBACKS
    matplotlib.rcParams["pdf.fonttype"] = 42  # embed as real text, not Type 3 bitmaps
    matplotlib.rcParams["ps.fonttype"] = 42


apply_rcparams()


def resolve_font(weight: str = "regular") -> str:
    """Return a font file path for the fallback list, warning once if Arial itself
    isn't installed (PLOS requires Arial, Times, or Symbol)."""
    path = font_manager.findfont(
        font_manager.FontProperties(family=FONT_FALLBACKS, weight=weight),
        fallback_to_default=True,
    )
    if "arial" not in Path(path).stem.lower():
        warnings.warn(
            f"Arial is not installed on this machine -- using '{Path(path).stem}' instead. "
            "PLOS requires Arial, Times, or Symbol; re-render on a machine with Arial "
            "installed before final submission.",
            stacklevel=2,
        )
    return path


def pt_to_px(pt: float, dpi: int = DPI) -> int:
    return round(pt / 72.0 * dpi)


def compress_eps(eps_path: Path, threshold_mb: float = 5.0) -> None:
    """Re-distill through Ghostscript: matplotlib/Pillow both write embedded-raster
    EPS uncompressed (can run 10s of MB for a full-panel composite), well over PLOS's
    10MB cap; eps2write's default image compression gets it down losslessly. Only
    bother for files already at real risk of the cap -- re-distilling a small, pure
    vector EPS (no embedded raster) adds Ghostscript's own overhead and makes it bigger."""
    if eps_path.stat().st_size <= threshold_mb * 1024 * 1024:
        return
    if shutil.which("gs") is None:
        warnings.warn(
            f"Ghostscript ('gs') not found -- {eps_path.name} may be an uncompressed "
            "raster EPS well over PLOS's 10MB limit. Install ghostscript and re-run, "
            "or submit the .tiff instead (PLOS accepts TIFF or EPS, not both).",
            stacklevel=2,
        )
        return
    compressed_path = eps_path.with_suffix(".compressed.eps")
    subprocess.run(
        ["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dEPSCrop", "-sDEVICE=eps2write",
         f"-sOutputFile={compressed_path}", str(eps_path)],
        check=True,
    )
    compressed_path.replace(eps_path)


def report_sizes(out_stem: Path, extra: str = "") -> None:
    sizes_kb = {ext: out_stem.with_suffix(f".{ext}").stat().st_size / 1024 for ext in FORMATS}
    suffix = f" ({extra})" if extra else ""
    print(f"[ok] {out_stem.name}{suffix}: " + ", ".join(f"{e}={kb:.0f}KB" for e, kb in sizes_kb.items()))


def save_all_formats(fig, out_stem: Path, dpi: int = DPI) -> None:
    """Write png/pdf/eps directly from a matplotlib figure, derive a true 8-bit RGB
    (no alpha channel) tiff from the png via PIL, and Ghostscript-compress the eps."""
    png_path = out_stem.with_suffix(".png")
    fig.savefig(png_path, dpi=dpi, facecolor="white")
    fig.savefig(out_stem.with_suffix(".pdf"), dpi=dpi, facecolor="white")
    fig.savefig(out_stem.with_suffix(".eps"), facecolor="white")

    with Image.open(png_path) as im:
        im.convert("RGB").save(
            out_stem.with_suffix(".tiff"), dpi=(dpi, dpi), compression="tiff_lzw"
        )
    compress_eps(out_stem.with_suffix(".eps"))
    report_sizes(out_stem)


def compose_side_by_side(
    panel_a_path: Path,
    panel_b_path: Path,
    dpi: int = DPI,
    border_pt: float = 2.0,
    label_pt: float = 12.0,
    labels: tuple = ("A", "B"),
) -> Image.Image:
    """Paste two equal-height panel rasters side by side with bold corner labels and
    a white border -- the shared final-assembly step for the paper's lettered figures."""
    im_a = Image.open(panel_a_path).convert("RGB")
    im_b = Image.open(panel_b_path).convert("RGB")
    if im_a.height != im_b.height:
        raise ValueError(
            f"Panels must have equal height to concatenate cleanly: "
            f"{panel_a_path.name}={im_a.height}px vs {panel_b_path.name}={im_b.height}px"
        )

    border_px = pt_to_px(border_pt, dpi)
    canvas = Image.new(
        "RGB",
        (im_a.width + im_b.width + 2 * border_px, im_a.height + 2 * border_px),
        "white",
    )
    canvas.paste(im_a, (border_px, border_px))
    canvas.paste(im_b, (border_px + im_a.width, border_px))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(resolve_font(weight="bold"), pt_to_px(label_pt, dpi))
    inset_px = pt_to_px(6.0, dpi)
    draw.text((border_px + inset_px, border_px + inset_px), labels[0], fill="black", font=font)
    draw.text((border_px + im_a.width + inset_px, border_px + inset_px), labels[1], fill="black", font=font)

    return canvas


def save_composite_all_formats(canvas: Image.Image, out_stem: Path, dpi: int = DPI) -> None:
    canvas.save(out_stem.with_suffix(".png"), dpi=(dpi, dpi))
    canvas.save(out_stem.with_suffix(".tiff"), dpi=(dpi, dpi), compression="tiff_lzw")

    fig = plt.figure(figsize=(canvas.width / dpi, canvas.height / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(np.asarray(canvas), interpolation="none")
    ax.axis("off")
    fig.savefig(out_stem.with_suffix(".pdf"), dpi=dpi)
    fig.savefig(out_stem.with_suffix(".eps"), dpi=dpi)
    plt.close(fig)
    compress_eps(out_stem.with_suffix(".eps"))

    report_sizes(out_stem, extra=f"{canvas.width}x{canvas.height}px @ {dpi}dpi")

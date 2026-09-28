from pathlib import Path

import matplotlib as mpl


def save_panel(
    fig,
    stem,
    *,
    formats=("svg", "pdf", "png"),
    dpi=600,
    pad_inches=0.1,
    svg_fonttype="path",
):
    """Save with one fixed crop for all formats, preserving the current layout.

    pad_inches=0.1 matches normal Matplotlib inline-preview padding. The crop
    includes visible annotations outside the axes, as bbox_inches="tight" did.
    Compute it once from the figure canvas so different export backends do not
    each choose a different crop. A layout engine, if present, is temporarily
    frozen after drawing, then restored. Figure size and data are not modified.

    svg_fonttype="path": vector glyph outlines; consistent font shapes.
    svg_fonttype="none": editable text; viewer font substitution is possible.

    The figure background is preserved. dpi controls raster export resolution,
    not the detail available in the original microscopy/point-cloud images.
    stem is a path without an extension. Returns output Paths in formats order.
    Requires a renderer-capable canvas, such as the normal notebook Agg canvas.
    """
    if pad_inches < 0:
        raise ValueError("pad_inches must be nonnegative.")
    if svg_fonttype not in ("path", "none"):
        raise ValueError("svg_fonttype must be 'path' or 'none'.")
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    # Resolve text/annotation extents and any layout engine at the preview DPI.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    crop = fig.get_tightbbox(renderer).padded(pad_inches).frozen()
    engine = fig.get_layout_engine()
    if engine is not None:
        fig.set_layout_engine("none")

    outputs = []
    try:
        with mpl.rc_context(
            {
                "svg.fonttype": svg_fonttype,
                "pdf.fonttype": 42,
                "savefig.bbox": None,
            }
        ):
            for extension in formats:
                path = stem.parent / f"{stem.name}.{extension}"
                fig.savefig(
                    path,
                    dpi=dpi,
                    facecolor=fig.get_facecolor(),
                    transparent=False,
                    bbox_inches=crop,  # The SAME explicit box in every format.
                    pad_inches=0,  # Padding is already included in crop.
                )
                outputs.append(path)
    finally:
        if engine is not None:
            fig.set_layout_engine(engine)
    return outputs

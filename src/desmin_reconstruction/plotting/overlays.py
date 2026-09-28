"""PyVista point-cloud renders arranged as Protein 1 / Protein2 / Overlay.

Pass the existing render_pointcloud helper from your notebook. It must add
points to the supplied Plotter and return that same Plotter (or None).
The three renders share one XY extent and camera. Matplotlib adds all labels,
axes and the overlay ROI rectangle after rendering.

Render once, then call compose_upper_row repeatedly to adjust the layout:

    rendered = render_upper_row(points, COLOR_DICT, render_pointcloud)
    fig, axes = compose_upper_row(rendered, roi=(0.5, 2.0, 10.0, 10.0))
    save_row(fig, "../figures/fig_1/1b/upper_row", svg_fonttype="path")

Requirements: numpy, matplotlib, pyvista, and a working VTK rendering backend.
"""
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter


@dataclass(frozen=True)
class RenderedRow:
    images: tuple
    extent: tuple  # Requested physical view: xmin, xmax, ymin, ymax.
    image_extent: tuple  # Actual camera view, accounting for pixel rounding.
    title_colors: tuple


def render_overlays(
    points,
    colors,
    render_pointcloud,
    *,
    protein_names=("desmin", "actinin"),
    xy_bounds=None,
    height_px=700,
    opacity=0.2,
    point_size=1.5,
):
    """Render two channels individually and together, without axes or ROI.

    xy_bounds is (xmin, xmax, ymin, ymax), in the same coordinates as the
    rendered points. None uses the combined scene's actual bounds, avoiding
    guesses from rounded tick labels. Supply exact acquisition bounds if the
    field of view extends beyond the outermost points. No data are rescaled.

    The existing helper receives opacity and point_size unchanged. For a helper
    that uses VTK pixel-sized points, scale point_size proportionally when
    increasing height_px to preserve the relative dot size in the final panel.

    Returned images are in [channel 1, channel 2, overlay] order. The overlay
    is rendered from both point clouds together, not from combined screenshots.
    """
    import pyvista as pv

    names = tuple(protein_names)
    if len(names) != 2:
        raise ValueError("protein_names must contain exactly two channels.")
    height_px = int(height_px)
    if height_px < 2:
        raise ValueError("height_px must be at least 2.")
    images = [None, None, None]
    p = pv.Plotter(off_screen=True, window_size=(height_px, height_px))
    try:
        # Build the overlay first to obtain common bounds for both channels.
        for output_index, channels in ((2, names), (0, names[:1]), (1, names[1:])):
            p.clear()
            p.set_background("black")
            for name in channels:
                result = render_pointcloud(
                    p,
                    points[name],
                    color=colors[name],
                    opacity=opacity,
                    point_size=point_size,
                )
                if result is not None and result is not p:
                    raise ValueError("render_pointcloud must use the supplied Plotter.")

            if output_index == 2:
                scene = np.asarray(p.bounds, dtype=float)
                if not np.all(np.isfinite(scene)) or scene[5] < scene[4]:
                    raise ValueError("The combined scene has no valid bounds.")
                extent = np.asarray(
                    scene[:4] if xy_bounds is None else xy_bounds, dtype=float
                )
                if extent.shape != (4,) or not np.all(np.isfinite(extent)):
                    raise ValueError("xy_bounds must contain four finite coordinates.")
                x0, x1, y0, y1 = extent
                dx, dy = x1 - x0, y1 - y0
                if dx <= 0 or dy <= 0:
                    raise ValueError("X and Y bounds must be increasing.")
                width_px = max(2, int(np.ceil(height_px * dx / dy)))
                p.window_size = (width_px, height_px)
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                cz = (scene[4] + scene[5]) / 2
                distance = 2 * max(dx, dy, scene[5] - scene[4])
                camera_position = [(cx, cy, cz + distance), (cx, cy, cz), (0, 1, 0)]
                # ceil() may add less than one pixel of width. Retain the true
                # camera extent so the Matplotlib coordinates remain accurate.
                half_width = (dy / 2) * width_px / height_px
                image_extent = (cx - half_width, cx + half_width, y0, y1)

            # Set XY view independently of whether an ROI box will be shown.
            p.enable_parallel_projection()
            p.camera_position = camera_position
            p.camera.parallel_scale = dy / 2
            p.reset_camera_clipping_range()
            p.render()
            images[output_index] = np.array(
                p.screenshot(transparent_background=False), copy=True
            )
        title_colors = tuple(pv.Color(colors[name]).float_rgb for name in names) + (
            "white",
        )
    finally:
        p.close()
    return RenderedRow(tuple(images), tuple(extent), image_extent, title_colors)


def compose_overlays(
    rendered,
    *,
    titles=tuple[str, str, str],
    roi:tuple[float, float, float, float]|None=None,
    panel_height=2.6,
    panel_gap=0.25,
    xlabel_fontsize=14,
    ylabel_fontsize=14,
    xtick_fontsize=10,
    ytick_fontsize=10,
    title_fontsize=14,
    title_pad=6,
    xtick_pad=6,
    ytick_pad=9,
    xlabel_pad=7,
    ylabel_pad=9,
    ytick_rotation=90,
    tick_format="%.1f",
    xticks=None,
    yticks=None,
    font_family="DejaVu Sans",
    roi_linewidth=0.8,
):
    """Compose the three screenshots with shared axes and figure-level labels.

    panel_height and panel_gap are in INCHES. All font sizes, other padding,
    and line widths are in POINTS. Decrease xlabel_pad to bring x (µm) closer
    to the X tick labels. Decrease panel_gap to bring panels closer together.

    roi is (left, bottom, width, height) in data coordinates, drawn on the
    overlay only. None hides it. The default corresponds to a square spanning
    X=0.5..10.5 and Y=2..12; no BoundingSquare object is needed.

    X and Y limits/ticks are linked. All panels show the Y axis and tick marks;
    only the first panel shows numeric Y tick labels.
    Endpoint label alignment prevents collisions at the origin and in gaps.
    """
    if len(rendered.images) != 3 or len(titles) != 3:
        raise ValueError("Exactly three images and titles are required.")
    if panel_height <= 0 or panel_gap < 0:
        raise ValueError(
            "panel_height must be positive; panel_gap must be nonnegative."
        )
    x0, x1, y0, y1 = rendered.extent
    dx, dy = x1 - x0, y1 - y0
    width = panel_height * dx / dy
    xticks = np.linspace(x0, x1, 4) if xticks is None else np.asarray(xticks)
    yticks = np.linspace(y0, y1, 4) if yticks is None else np.asarray(yticks)
    left = max(
        0.65,
        (ytick_pad + ylabel_pad + 1.3 * (ytick_fontsize + ylabel_fontsize) + 8) / 72,
    )
    bottom = max(
        0.6,
        (xtick_pad + xlabel_pad + 1.3 * (xtick_fontsize + xlabel_fontsize) + 8) / 72,
    )
    top, right = max(0.35, (1.3 * title_fontsize + title_pad + 6) / 72), 0.12
    total_width = 3 * width + 2 * panel_gap
    fw, fh = left + total_width + right, bottom + panel_height + top

    with mpl.rc_context({"font.family": font_family}):
        fig = plt.figure(figsize=(fw, fh), facecolor="black")
        axes = []
        for i, (rgb, title, title_color) in enumerate(
            zip(rendered.images, titles, rendered.title_colors)
        ):
            shared = {} if i == 0 else {"sharex": axes[0], "sharey": axes[0]}
            ax = fig.add_axes(
                [
                    (left + i * (width + panel_gap)) / fw,
                    bottom / fh,
                    width / fw,
                    panel_height / fh,
                ],
                facecolor="black",
                **shared,
            )
            axes.append(ax)
            ax.imshow(
                rgb,
                extent=rendered.image_extent,
                origin="upper",
                interpolation="none",
                aspect="equal",
            )
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            # Set limits AFTER tick placement: set_ticks can expand limits.
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.xaxis.set_major_formatter(FormatStrFormatter(tick_format))
            ax.yaxis.set_major_formatter(FormatStrFormatter(tick_format))
            ax.tick_params(colors="white", direction="in", length=3, width=0.7)
            ax.tick_params(axis="x", labelsize=xtick_fontsize, pad=xtick_pad)
            # Hide labels locally; do not replace the shared Y formatter.
            ax.tick_params(
                axis="y",
                left=True,
                labelleft=(i == 0),
                labelsize=ytick_fontsize,
                pad=ytick_pad,
                labelrotation=ytick_rotation,
            )
            for side, spine in ax.spines.items():
                spine.set_visible(side in ("left", "bottom"))
                spine.set_color("white")
                spine.set_linewidth(0.7)
            ax.set_title(
                title,
                color=title_color,
                fontsize=title_fontsize,
                pad=title_pad,
                weight="semibold",
            )
            for label in ax.get_xticklabels():
                label.set_ha("center")
            for value, label in zip(ax.get_yticks(), ax.get_yticklabels()):
                label.set_va("bottom" if np.isclose(value, y0) else "center")

        if roi is not None:
            rx, ry, rw, rh = roi
            axes[2].add_patch(
                Rectangle(
                    (rx, ry),
                    rw,
                    rh,
                    fill=False,
                    edgecolor="white",
                    linewidth=roi_linewidth,
                    clip_on=False,
                    zorder=5,
                )
            )

        # Measure labels so the common titles have explicit, useful padding.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        xboxes = [
            t.get_window_extent(renderer)
            for ax in axes
            for t in ax.get_xticklabels()
            if t.get_visible()
        ]
        yboxes = [
            t.get_window_extent(renderer)
            for t in axes[0].get_yticklabels()
            if t.get_visible()
        ]
        inverse = fig.transFigure.inverted()
        xlabel_y = inverse.transform((0, min(b.y0 for b in xboxes)))[1] - xlabel_pad / (
            72 * fh
        )
        ylabel_x = inverse.transform((min(b.x0 for b in yboxes), 0))[0] - ylabel_pad / (
            72 * fw
        )
        fig.supxlabel(
            "x (µm)",
            x=(left + total_width / 2) / fw,
            y=xlabel_y,
            va="top",
            color="white",
            fontsize=xlabel_fontsize,
        )
        fig.supylabel(
            "y (µm)",
            x=ylabel_x,
            y=(bottom + panel_height / 2) / fh,
            ha="right",
            color="white",
            fontsize=ylabel_fontsize,
        )
    return fig, tuple(axes)

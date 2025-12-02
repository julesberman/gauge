import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Tuple, Union

import imageio  # pip install "imageio[ffmpeg]"
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from einops import rearrange
from IPython.display import HTML
from matplotlib import animation, colors
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable


def imshow_movie(
    sol: np.ndarray,
    *,
    frames: Optional[int] = 50,
    t: Optional[Sequence[float]] = None,
    interval: int = 100,
    title: str = "",
    cmap: str = "viridis",
    aspect: str = "equal",
    interpolation: str = "nearest",
    figsize: Optional[Tuple[float, float]] = None,        # ← NEW
    show_colorbar: bool = True,
    live_cbar: bool = True,
    tight: bool = True,
    c_norm: Optional[Tuple[float, float]] = None,
    t_txt: bool = True,
    gif_hq: bool = False,
    #
    save_to: Optional[Union[str, Path]] = None,
    save_format: Literal["gif", "mp4"] = "gif",
    fps: int = 10,
    dpi: Optional[int] = None,
    #
    show_inline: bool = True,
) -> Optional[HTML]:
    """
    Animate a stack of 2-D images (T × H × W).

    Parameters
    ----------
    sol : ndarray
        Array of frames ordered in time, shape (T, H, W).
    frames : int | None
        Number of frames to *display* (sub-samples if needed, None ⇒ all).
    t : sequence[float] | None
        Time-stamps for each frame (defaults to np.arange(T)).
    interval : int
        Delay between frames in ms for the JS/HTML player.
    figsize : (w, h) in inches | None
        Figure size passed to `plt.subplots`; None uses Matplotlib default.
    show_colorbar : bool
        Draw a colour-bar (independent of live updates).
    live_cbar : bool
        If True, rescale colour limits every frame.
    tight : bool
        Remove all extra padding/margins around the axes.
    c_norm : (vmin, vmax) | None
        Fixed colour limits (overrides data-driven limits).
    t_txt : bool
        Show time/frame information in the axes title.
    save_to : str | Path | None
        Persist animation to disk if not None (extension added).
    save_format : {"gif", "mp4"}
        Container used when saving.
    fps : int
        Frames-per-second for the saved video.
    dpi : int | None
        DPI for MP4 writer; ignored for GIF.
    show_inline : bool
        Return an IPython HTML widget (JS) for inline display.

    Returns
    -------
    IPython.display.HTML | None
        Inline HTML widget if `show_inline` else None.
    """
    sol = np.asarray(sol)
    T = sol.shape[0]

    # Validate/build time vector
    if t is None:
        t = np.arange(T)
    else:
        t = np.asarray(t)
        if t.shape[0] != T:
            raise ValueError("`t` must have the same length as sol.shape[0]")

    # Frame sub-sampling
    step = 1 if frames is None else max(T // frames, 1)
    sol_frames, t_frames = sol[::step], t[::step]

    # Figure & axes
    fig, ax = plt.subplots(figsize=figsize)
    cax = None
    if show_colorbar:
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad="3%")

    norm = (
        colors.Normalize(*c_norm)
        if c_norm is not None
        else colors.Normalize(vmin=float(sol.min()), vmax=float(sol.max()))
    )
    im = ax.imshow(
        sol_frames[0],
        cmap=cmap,
        aspect=aspect,
        interpolation=interpolation,
        norm=norm,
    )
    if show_colorbar:
        _ = fig.colorbar(im, cax=cax)  # noqa: F841

    ax.set_xticks([])
    ax.set_yticks([])
    tx = ax.set_title("") if t_txt else None

    # Remove padding if tight=True
    if tight:
        ax.margins(0)
        fig.tight_layout(pad=0)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Animation callback
    def _animate(idx: int):
        arr = sol_frames[idx]
        im.set_data(arr)
        if live_cbar and show_colorbar:
            im.set_clim(float(arr.min()), float(arr.max()))
        if t_txt:
            tx.set_text(f"{title} t={t_frames[idx]:.3g} (frame {idx})")

    ani = animation.FuncAnimation(
        fig,
        _animate,
        frames=len(sol_frames),
        interval=interval,
        blit=False,
    )

    # Save to disk
    if save_to is not None:
        path = Path(save_to).with_suffix(f".{save_format}")
        if save_format == "gif":
            if gif_hq:
                _save_gif_high_quality(ani, path, fps=fps, dpi=dpi or 100)
            else:
                writer = animation.PillowWriter(fps=fps)
                ani.save(path, writer=writer)
        else:  # mp4
            writer = animation.FFMpegWriter(fps=fps, codec="libx264")
            ani.save(path, writer=writer, dpi=dpi)
        print(f"animation saved → {path.resolve()}")

    plt.close(fig)
    return HTML(ani.to_jshtml()) if show_inline else None


def _save_gif_high_quality(ani, path: Path, fps: int, dpi: int | None):
    """Write *path* as a palette-optimised GIF via ffmpeg."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_mp4 = Path(tmp, "tmp.mp4")
        palette = Path(tmp, "palette.png")

        # 1) MP4 (true colour, no palette limits)
        writer = animation.FFMpegWriter(fps=fps, codec="libx264")
        ani.save(tmp_mp4, writer=writer, dpi=dpi)

        # 2) Generate optimal palette
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y",
             "-i", tmp_mp4, "-vf", "palettegen=stats_mode=diff", palette],
            check=True,
        )
        # 3) Apply palette with Floyd–Steinberg dithering
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y",
             "-i", tmp_mp4, "-i", palette,
             "-lavfi", "paletteuse=dither=floyd_steinberg",
             "-loop", "0", path],
            check=True,
        )


def imshow_pts_movies(
    sol,
    pts,
    extent,
    c="r",
    size=None,
    alpha=1,
    frames=50,
    t=None,
    interval=100,
    tight=False,
    title="",
    cmap="viridis",
    aspect="equal",
    live_cbar=False,
    save_to=None,
    show=True,
    fps=30,
):

    fig, ax = plt.subplots()
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", "5%", "5%")

    cv0 = sol[0]
    # Here make an AxesImage rather than contour
    im = ax.imshow(cv0, cmap=cmap, aspect=aspect, extent=extent)
    sct = ax.scatter(x=pts[0, 0], y=pts[0, 1], alpha=alpha, s=size, c=c)
    cb = fig.colorbar(im, cax=cax)
    tx = ax.set_title("Frame 0")
    vmax = np.max(sol)
    vmin = np.min(sol)

    if tight:
        plt.tight_layout()

    def animate(frame):
        (arr, scatter), t = frame
        im.set_data(arr)
        im.set_extent(extent)
        sct.set_offsets(scatter.T)
        if live_cbar:
            vmax = np.max(arr)
            vmin = np.min(arr)
            im.set_clim(vmin, vmax)
        tx.set_text(f"{title} t={t:.2f}")

    time, w, h = sol.shape
    if t is None:
        t = np.arange(time)
    inc = max(time // frames, 1)
    sol_frames = sol[::inc]
    t_frames = t[::inc]
    pts = pts[::inc]
    sol_frames = zip(sol_frames, pts, strict=False)
    frames = list(zip(sol_frames, t_frames, strict=False))
    ani = FuncAnimation(
        fig,
        animate,
        frames=frames,
        interval=interval,
    )
    plt.close()

    if save_to is not None:
        p = Path(save_to).with_suffix(".gif")
        ani.save(p, writer="pillow", fps=fps)

    if show:
        return HTML(ani.to_jshtml())


def scatter_movie(
    pts,
    c="r",
    n_samples=None,
    size=None,
    xlim=None,
    ylim=None,
    alpha=1,
    frames=60,
    t=None,
    title="",
    interval=100,
    save_to=None,
    show=True,
    fps=10,
):
    pts = np.asarray(pts)

    if len(pts.shape) == 4:
        g, _, n, _ = pts.shape
        c = []
        colors = ["r", "b", "g", "m", "k"]
        for i in range(g):
            c.extend([colors[i]] * n)
        pts = rearrange(pts, "g t n d -> t (g n) d")

    pts = rearrange(pts, "t n d -> t d n")

    if n_samples is not None:
        sample_idx = np.random.choice(
            pts.shape[-1] - 1, size=n_samples, replace=False)
        sample_idx = np.asarray(sample_idx, dtype=np.int32)
        print(sample_idx, pts.shape)
        pts = pts[:, :, sample_idx]
        print(pts.shape)
        if type(c) == list:
            c = c[sample_idx]
    fig, ax = plt.subplots()

    sct = ax.scatter(x=pts[0, 0], y=pts[0, 1],
                     alpha=alpha, s=size, c=c)
    mm = pts.min(axis=(0, 2))
    mx = pts.max(axis=(0, 2))

    if xlim is None:
        xlim = [mm[0], mx[0]]
    if ylim is None:
        ylim = [mm[1], mx[1]]
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    tx = ax.set_title("Frame 0")

    def animate(frame):
        scatter, t = frame
        sct.set_offsets(scatter.T)
        tx.set_text(f"{title} t={t:.2f}")

    time = len(pts)
    if t is None:
        t = np.arange(time)
    inc = max(time // frames, 1)
    t_frames = t[::inc]
    pts = pts[::inc]

    frames = list(zip(pts, t_frames, strict=False))
    ani = FuncAnimation(
        fig,
        animate,
        frames=frames,
        interval=interval,
    )
    plt.close()

    if save_to is not None:
        p = Path(save_to).with_suffix(".gif")
        ani.save(p, writer="pillow", fps=fps)

    if show:
        return HTML(ani.to_jshtml())


def line_movie(
    sol,
    frames=50,
    t=None,
    x=None,
    color=None,
    title="",
    interval=100,
    ylim=None,
    save_to=None,
    show=True,
    legend=None,
    tight=False,
    fps=10,
):
    sol = np.asarray(sol)
    if len(sol.shape) == 2:
        sol = np.expand_dims(sol, axis=0)

    n_lines, time, space = sol.shape
    sol = rearrange(sol, "l t s -> t s l")
    fig, ax = plt.subplots()
    ax.set_ylim([sol.min(), sol.max()])
    if ylim is not None:
        ax.set_ylim(ylim)
    if x is None:
        x = np.arange(sol.shape[1])

    if color is not None:
        cycler = plt.cycler(color=color)
        ax.set_prop_cycle(cycler)
    line = ax.plot(
        x,
        sol[0],
    )
    if tight:
        plt.tight_layout()

    if legend is not None:
        ax.legend(legend)

    def animate(frame):
        sol, t = frame
        ax.set_title(f"{title} t={t:.3f}")
        for i, l in enumerate(line):
            l.set_ydata(sol[:, i])
        return line

    def init():
        line.set_ydata(np.ma.array(x, mask=True))
        return (line,)

    if t is None:
        t = np.arange(time)
    inc = max(time // frames, 1)
    sol_frames = sol[::inc]
    t_frames = t[::inc]
    sol_frames = sol[::inc]
    frames = list(zip(sol_frames, t_frames, strict=False))
    ani = FuncAnimation(fig, animate, frames=frames,
                        interval=interval, blit=True)
    plt.close()
    if save_to is not None:
        p = Path(save_to).with_suffix(".gif")
        ani.save(p, writer="pillow", fps=fps)

    if show:
        return HTML(ani.to_jshtml())


def trajectory_movie(
    y,
    frames=50,
    title="",
    ylabel="",
    xlabel="Time",
    legend=[],
    x=None,
    interval=100,
    ylim=None,
    save_to=None,
):

    y = np.asarray(y)
    if x is None:
        x = np.arange(len(y))

    fig, ax = plt.subplots()
    total = len(x)
    inc = max(total // frames, 1)
    x = x[::inc]
    y = y[::inc]
    if ylim is None:
        ylim = np.array([y.min(), y.max()])
    xlim = [x.min(), x.max()]

    def animate(i):
        ax.cla()
        ax.plot(x[:i], y[:i], marker="o", markevery=[-1])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend(legend, loc="lower right")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} t={x[i]:.2f}")

    ani = FuncAnimation(fig, animate, frames=len(x), interval=interval)
    plt.close()

    if save_to is not None:
        p = Path(save_to).with_suffix(".gif")
        ani.save(p, writer="pillow", fps=30)

    return HTML(ani.to_jshtml())


def plot_grid(
    A,
    *,
    fig=None,                 # NEW: existing Figure
    colorbar=True,
    colorbar_mode="single",
    grid_height=None,
    grid_width=None,
    fig_size=(8, 8),
    cmap="viridis",
    xticks_on=False,
    yticks_on=False,
    aspect="auto",
    space=0.1,
    save_to=None,
    titles_x=None,
    titles_y=None,
    c_norm=None,
    to_int=False,
    cbar_ticks=None,
    cbar_tick_fmt=None,
    imagegrid_kwargs=None,
    imshow_kwargs=None,
    show=True
):
    """
    Display *N* images in a tidy rectangular grid.

    Parameters
    ----------
    A : array-like, shape (N, H, W) or (N, H, W, C)
        Stack of images to plot (first axis is the grid index).
        If *to_int* is ``True`` the data are clipped to ``[0, 255]`` and
        cast to ``uint16`` before display.
    fig : :class:`matplotlib.figure.Figure`, optional
        Existing figure to draw in.  A new one is created when ``None``.
    colorbar : bool, default ``True``
        Draw colourbar(s) alongside the images.
    colorbar_mode : {'single', 'each'}, default ``'single'``
        One shared colourbar for the whole grid or one per image.
    grid_height, grid_width : int or ``None``, optional
        Fix the number of rows/columns.  Any ``None`` dimension is inferred
        so that ``grid_height × grid_width ≥ N``.
    fig_size : (float, float), default ``(8, 8)``
        Size of the Matplotlib figure in inches.
    cmap : str or :class:`matplotlib.colors.Colormap`, default ``'viridis'``
        Colormap fed to :py:meth:`matplotlib.axes.Axes.imshow`.
    xticks_on, yticks_on : bool, default ``False``
        Toggle axis tick visibility.
    aspect : {'auto', 'equal'} or float, default ``'auto'``
        Aspect ratio argument passed to the internal ``ImageGrid``.
    space : float, default ``0.1``
        Normalised gap between grid cells (``axes_pad``).
    save_to : str or :class:`pathlib.Path`, optional
        File path; if given, the figure is saved via :pyfunc:`plt.savefig`.
    titles_x : Sequence[str], optional
        Column titles – *length must equal* ``grid_width``.
    titles_y : Sequence[str], optional
        Row titles – *length must equal* ``grid_height``.
    c_norm : (float, float) or ``None``, optional
        ``(vmin, vmax)`` shared by all panels.  ``None`` ⇒ auto-scale each.
    to_int : bool, default ``False``
        Clip data to 0–255 and convert to ``uint16`` before plotting.
    cbar_ticks : 1-D array-like, optional
        Explicit tick locations for the colourbar(s).
    cbar_tick_fmt : str or Callable, optional
        ``'%.1f'``-style format string **or** a ``FuncFormatter``-like
        callable for colourbar major ticks.
    imagegrid_kwargs : dict, optional
        Extra keyword arguments forwarded to
        :class:`mpl_toolkits.axes_grid1.ImageGrid`.
    imshow_kwargs : dict, optional
        Extra keyword arguments forwarded to each ``ax.imshow`` call.
    show : bool, default ``True``
        Call :pyfunc:`plt.show` immediately.  When ``False`` the function
        returns the created :class:`ImageGrid` for further tweaking.

    Returns
    -------
    None or :class:`mpl_toolkits.axes_grid1.ImageGrid`
        * ``None`` if *show* is ``True`` (figure already displayed).
        * The ``ImageGrid`` instance if *show* is ``False``.
    """
    if to_int:
        A = np.clip(A, a_min=0, a_max=255)
        A = np.asarray(A, dtype=np.uint16)

    # ---- resolve grid shape -------------------------------------------------
    N = A.shape[0]
    if grid_height is None and grid_width is None:
        grid_width = int(np.ceil(np.sqrt(N)))
        grid_height = int(np.ceil(N / grid_width))
    elif grid_height is None:
        grid_height = int(np.ceil(N / grid_width))
    elif grid_width is None:
        grid_width = int(np.ceil(N / grid_height))

    # ---- validate titles ----------------------------------------------------
    if titles_x is not None and len(titles_x) != grid_width:
        raise ValueError(
            f"Expected {grid_width} column titles, got {len(titles_x)}.")
    if titles_y is not None and len(titles_y) != grid_height:
        raise ValueError(
            f"Expected {grid_height} row titles, got {len(titles_y)}.")

    # ---- prepare extra kwargs -----------------------------------------------
    imagegrid_kwargs = imagegrid_kwargs or {}
    imshow_kwargs = imshow_kwargs or {}

    # ---- create or reuse Figure ---------------------------------------------
    if fig is None:
        fig = plt.figure(figsize=fig_size)

    # ---- create ImageGrid ---------------------------------------------------
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(grid_height, grid_width),
        axes_pad=space,
        share_all=True,
        cbar_mode=(colorbar_mode if colorbar else None),
        aspect=aspect,
        **imagegrid_kwargs,
    )

    # ---- plot each image ----------------------------------------------------
    for i in range(N):
        ax = grid[i]

        norm = (colors.Normalize(vmin=c_norm[0], vmax=c_norm[1])
                if c_norm is not None
                else colors.Normalize(vmin=np.min(A[i]), vmax=np.max(A[i])))

        im = ax.imshow(
            A[i],
            cmap=cmap,
            aspect="auto",
            norm=norm,
            **imshow_kwargs,
        )

        if not xticks_on:
            ax.set_xticks([])
        if not yticks_on:
            ax.set_yticks([])

        if colorbar and colorbar_mode == "each":
            cax = ax.cax
            cax.colorbar(im)
            cax.tick_params(labelleft=True)

    # ---- shared colorbar ----------------------------------------------------
    if colorbar and colorbar_mode == "single":
        cax = grid.cbar_axes[0]
        if cbar_ticks is not None:
            cbar = cax.colorbar(im, ticks=cbar_ticks)
        else:
            cbar = cax.colorbar(im)
        cax.tick_params()
        if cbar_tick_fmt is not None:
            if type(cbar_tick_fmt) != "str":
                cbar.ax.yaxis.set_major_formatter(FuncFormatter(cbar_tick_fmt))
            else:
                cbar.ax.yaxis.set_major_formatter(
                    FormatStrFormatter(cbar_tick_fmt))
    # ---- column titles ------------------------------------------------------
    if titles_x is not None:
        for col in range(grid_width):
            grid[col].set_title(titles_x[col])

    # ---- row titles ---------------------------------------------------------
    if titles_y is not None:
        for row in range(grid_height):
            idx = row * grid_width
            grid[idx].set_ylabel(
                titles_y[row],
                rotation=0,
                ha="right",
                va="center",
                labelpad=10,
            )

    # ---- save or show -------------------------------------------------------
    if save_to is not None:
        plt.savefig(Path(save_to))

    if show:
        plt.show()
        return
    else:
        return grid


def plot_grid_movie(

    A,
    t=None,
    titles_x=None,
    titles_y=None,
    suptitle=None,
    suptitle_y=None,
    colorbar=True,
    colorbar_mode="single",
    grid_height=None,
    grid_width=None,
    fig_size=(8, 8),
    cmap="viridis",
    xticks_on=False,
    yticks_on=False,
    aspect="auto",
    space=0.1,
    interval=200,
    save_to=None,
    show=True,
    live_cbar=True,
    c_norm=None,
    seconds=5,
    writer="pillow",
    frames=64,
    to_int=False,
):
    """
    Animate a grid of *N* miniature movies and return an embeddable HTML
    representation (or save as a GIF).

    Parameters
    ----------
    A : array-like, shape ``(N, T, H, W)`` or ``(N, T, H, W, C)``
        Stack of *N* movies, each with *T* time-steps.
        When *to_int* is ``True`` the data are clipped to ``[0, 255]``
        and cast to ``uint16`` before display.
    t : 1-D array-like or ``None``, optional
        Time stamps for each frame (currently unused – reserved).
    titles_x : Sequence[str], optional
        Column titles – length **must equal** *grid_width* if supplied.
    titles_y : Sequence[str], optional
        Row titles – length **must equal** *grid_height* if supplied.
    suptitle : str, optional
        Figure-level title.
    suptitle_y : float, optional
        Vertical position (in figure fraction) for *suptitle*.
    colorbar : bool, default ``True``
        Draw colourbar(s).
    colorbar_mode : {'single', 'each'}, default ``'single'``
        One shared colourbar or one per panel.
    grid_height, grid_width : int or ``None``, optional
        Fix grid size; any ``None`` side is inferred so that
        ``grid_height × grid_width ≥ N``.  
        If **both** are given, only the first
        ``grid_height × grid_width`` movies are drawn.
    fig_size : (float, float), default ``(8, 8)``
        Figure size in inches.
    cmap : str or :class:`matplotlib.colors.Colormap`, default ``'viridis'``
    xticks_on, yticks_on : bool, default ``False``
        Toggle axis tick visibility.
    aspect : {'auto', 'equal'} or float, default ``'auto'``
        Aspect ratio argument passed to the internal ``ImageGrid``.
    space : float, default ``0.1``
        Normalised padding between grid cells (``axes_pad``).
    interval : int, default ``200``
        Delay between frames *in milliseconds* for the animation.
    save_to : str or :class:`pathlib.Path`, optional
        Base name to save the animation as ``<name>.gif`` using *writer*.
        Ignored if ``None``.
    show : bool, default ``True``
        Return an embeddable :class:`IPython.display.HTML` object.  
        If ``False`` the function performs the save (if requested) and
        returns ``None``.
    live_cbar : bool, default ``True``
        Update colour limits every frame to reflect current data range.
    c_norm : (float, float) or ``None``, optional
        ``(vmin, vmax)`` applied to **all** frames. ``None`` ⇒ auto‐scale.
    seconds : int, default ``5``
        Target animation length in seconds (used to set FPS).
    writer : str, default ``'pillow'``
        Animation writer passed to Matplotlib (e.g. ``'imagemagick'``).
    frames : int, default ``64``
        Number of frames uniformly sampled from the input *T* steps.
    to_int : bool, default ``False``
        Clip data to 0–255 and convert to ``uint16`` before plotting.
    """
    A = np.asarray(A)
    if grid_width is not None and grid_height is not None:
        A = A[: grid_height * grid_width]
    # A is expected to be an array of movies with shape (n, t, h, w)
    t_idx = np.linspace(0, A.shape[1] - 1, frames, dtype=np.int32)
    A = A[:, t_idx]
    n, t = A.shape[:2]

    # Calculate grid dimensions if not provided
    if grid_height is None and grid_width is None:
        grid_width = int(np.ceil(np.sqrt(n)))
        grid_height = int(np.ceil(n / grid_width))
    elif grid_height is None:
        grid_height = int(np.ceil(n / grid_width))
    elif grid_width is None:
        grid_width = int(np.ceil(n / grid_height))

    # Compute vmin and vmax for consistent color scales
    vmin = A.min()
    vmax = A.max()

    # Create figure
    fig = plt.figure(figsize=fig_size)
    if suptitle is not None:
        fig.suptitle(suptitle, y=suptitle_y)

    # Set up image grid with specified aspect ratio, colorbar mode, and spacing
    cbar_mode = colorbar_mode if colorbar else None
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(grid_height, grid_width),
        axes_pad=space,
        share_all=True,
        cbar_mode=cbar_mode,
        aspect=aspect == "auto",
    )

    images = []

    if to_int:
        A = np.clip(A, a_min=0, a_max=255)
        A = np.asarray(A, dtype=np.uint16)

    # Plot initial images
    for i in range(n):
        ax = grid[i]
        if c_norm is not None:
            norm = colors.Normalize(vmin=c_norm[0], vmax=c_norm[1])
        else:
            norm = colors.Normalize(vmin=np.min(A[i, 0]), vmax=np.max(A[i, 0]))
        im = ax.imshow(A[i, 0], cmap=cmap, aspect="auto", norm=norm)
        images.append(im)
        if not xticks_on:
            ax.set_xticks([])
        if not yticks_on:
            ax.set_yticks([])

        # Add colorbar for each image if needed
        if colorbar and colorbar_mode == "each":
            cbar = ax.cax.colorbar(im)
            ax.cax.tick_params(labelleft=True)

        if titles_x is not None and i < grid_width:
            ax.set_title(titles_x[i])

        if titles_y is not None and i % grid_width == 0:
            ax.set_ylabel(titles_y.pop())

    # Add single colorbar if needed
    if colorbar and colorbar_mode == "single":
        cbar = grid.cbar_axes[0].colorbar(im)
        grid.cbar_axes[0].tick_params()

    # Define update function
    def update(frame):
        for i, im in enumerate(images):
            cur = A[i, frame]
            im.set_data(cur)
            if live_cbar:
                vmax = np.max(cur)
                vmin = np.min(cur)
                im.set_clim(vmin, vmax)
        return images

    # Create animation
    ani = FuncAnimation(fig, update, frames=t, interval=interval, blit=False)

    plt.close()
    if save_to is not None:
        p = Path(save_to).with_suffix(".gif")
        fps = t // seconds
        ani.save(
            p,
            writer=writer,
            fps=fps,
        )

    if show:
        return HTML(ani.to_jshtml())


def save_tensor_to_mp4(
    video: np.ndarray,
    out_path: Union[str, Path],
    *,
    fps: int = 30,
    seconds: float | None = None,                 # overrides fps if provided
    codec: str = "libx264",
    crf: int | None = 18,                    # constant-rate factor (0-51)
    bitrate: str | None = None,              # mutually exclusive with *crf*
    pix_fmt: str = "yuv420p",
    resize_to: Tuple[int, int] | None = None,
    progress: bool = True,
    cmap: str | mcolors.Colormap | None = None,   # NEW
    c_norm: Tuple[float, float] | mcolors.Normalize | None = None  # NEW
) -> None:
    """
    Encode a video tensor as an **H.264 MP4** file.

    Parameters
    ----------
    video : np.ndarray, shape ``(T, H, W, C)`` *or* ``(T, H, W)``
        Input frames.  
        *C* (channels) must be 1, 3, or 4.  
        *dtype* may be ``uint8`` (0–255) **or** ``float32/64``  
        in either **[0, 1]** (auto-scaled to 0–255) or **[0, 255]**.
    out_path : str or pathlib.Path
        Destination filename – the ``.mp4`` suffix is appended if missing.
    fps : int, default ``30``
        Frames per second.  Ignored when *seconds* is supplied.
    seconds : float, optional
        Target clip length in seconds.  Sets  
        ``fps = max(round(T / seconds), 1)``.
    codec : str, default ``'libx264'``
        FFmpeg codec name (any encoder supported by your FFmpeg build).
    crf : int, default ``18``
        Constant–rate factor **0 (best) – 51 (worst)**.  
        *Ignored if* *bitrate* is given.
    bitrate : str, optional
        Fixed bitrate such as ``'4M'``.  Mutually exclusive with *crf*.
    pix_fmt : str, default ``'yuv420p'``
        Pixel format handed to FFmpeg (use ``'yuv444p'`` for lossless RGBA).
    resize_to : (int, int), optional
        ``(height, width)`` to isotropically downsample every frame
        via OpenCV *INTER_AREA* before encoding.
    progress : bool, default ``True``
        Show textual progress bars using *tqdm*.
    cmap : str or matplotlib.colors.Colormap, optional
        Apply a Matplotlib colormap **before** encoding.  
        Requires **single-channel** input; results in 3-channel RGB output.
    c_norm : (vmin, vmax) tuple or matplotlib.colors.Normalize, optional
        Normalisation used together with *cmap*.  
        ``None`` ⇒ identity (no scaling).
    """

    # ---------- validation --------------------------------------------------
    if video.ndim not in (3, 4):
        raise ValueError("video must have shape (T, H, W[, C])")

    # Auto-expand (T, H, W) → (T, H, W, 1)
    if video.ndim == 3:
        video = video[..., None]

    if video.shape[-1] not in (1, 3, 4):
        raise ValueError("channel count must be 1, 3, or 4")

    if cmap is not None and video.shape[-1] != 1:
        raise ValueError("cmap can only be used with single-channel input")

    if not (np.issubdtype(video.dtype, np.floating) or video.dtype == np.uint8):
        raise TypeError("dtype must be float32/64 or uint8")

    T = video.shape[0]
    if seconds is not None:
        if seconds <= 0:
            raise ValueError("seconds must be positive")
        fps = max(int(round(T / seconds)), 1)

    # ---------- optional resize --------------------------------------------
    if resize_to is not None:
        import cv2  # pip install opencv-python-headless
        h_new, w_new = resize_to
        iterable = tqdm.tqdm(video, desc="Resizing") if progress else video
        video = np.stack(
            [cv2.resize(f, (w_new, h_new), interpolation=cv2.INTER_AREA)
             for f in iterable],
            axis=0,
        )

    # ---------- apply colormap (if requested) ------------------------------
    if cmap is not None:
        # Resolve cmap → Colormap instance
        cmap_obj = mcm.get_cmap(cmap) if isinstance(cmap, str) else cmap

        # Resolve c_norm → Normalize instance or identity
        if c_norm is None:
            def norm(x): return x
        elif isinstance(c_norm, tuple) or isinstance(c_norm, list):
            norm = mcolors.Normalize(vmin=c_norm[0], vmax=c_norm[1])
        else:
            norm = c_norm  # already a Normalize instance

        iterable = tqdm.tqdm(
            video, desc="Applying cmap") if progress else video
        recoloured = []
        for frame in iterable:
            frame_2d = frame[..., 0].astype(np.float32)
            # (H, W, 4), float in [0,1]
            rgba = cmap_obj(norm(frame_2d))
            rgb = (rgba[..., :3] * 255 + 0.5).astype(np.uint8)
            recoloured.append(rgb)
        video = np.stack(recoloured, axis=0)

    # ---------- float → uint8 *if needed* -----------------------------------
    if np.issubdtype(video.dtype, np.floating):
        # Detect if data are already 0–255: treat values > 1 as “already scaled”
        needs_scaling = video.max() <= 1.0
        if needs_scaling:
            if progress:
                print("Converting float32/64 in [0, 1] → uint8 …")
            video = (np.clip(video, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)
        else:
            if progress:
                print("Casting float32/64 in [0, 255] → uint8 (no rescale) …")
            video = video.round().astype(np.uint8)

    # ---------- assemble writer kwargs -------------------------------------
    writer_kwargs: dict[str, Any] = {
        "format": "ffmpeg",
        "mode": "I",
        "fps": fps,
        "codec": codec,
        "pixelformat": pix_fmt,
        "ffmpeg_log_level": "error",
    }

    # bitrate OR crf (mutually exclusive)
    if bitrate is not None:
        writer_kwargs["bitrate"] = bitrate
    elif crf is not None:
        writer_kwargs["ffmpeg_params"] = ["-crf", str(crf)]

    out_path = Path(out_path).with_suffix(".mp4")

    # ---------- encode ------------------------------------------------------
    with imageio.get_writer(out_path, **writer_kwargs) as writer:
        iterable = tqdm.tqdm(video, desc="Encoding",
                             unit="frame") if progress else video
        for frame in iterable:
            writer.append_data(frame)

    if progress:
        print(
            f"✓ Saved {len(video)} frames at {fps} fps → {out_path.resolve()}")


def plot_moments(true,
                 pred,
                 t=None,
                 show_samples=False,
                 true_var_style=None,
                 pred_var_style=None,
                 true_mean_style=None,
                 pred_mean_style=None,
                 figsize=(6, 4),
                 show=True,
                 save_to=None):
    """
    Plot mean and variance bands for two datasets (true and pred)
    over time-like axis t.

    Parameters
    ----------
    true : array-like, shape (n_samples_true, T)
    pred : array-like, shape (n_samples_pred, T)
    t    : array-like, shape (T,), optional
        If None, uses np.linspace(0, 1, T).
    """

    true = np.asarray(true)
    pred = np.asarray(pred)

    if true.shape[1] != pred.shape[1]:
        raise ValueError(
            f"`true` and `pred` must have the same second dimension (T). "
            f"Got {true.shape[1]} and {pred.shape[1]}."
        )

    T = true.shape[1]

    if t is None:
        t = np.linspace(0, 1, T)

    # Default styles (avoids mutable dicts in signature)
    if true_var_style is None:
        true_var_style = dict(alpha=0.25)
    if pred_var_style is None:
        pred_var_style = dict(alpha=0.25)
    if true_mean_style is None:
        true_mean_style = dict(color='C0', lw=2)
    if pred_mean_style is None:
        pred_mean_style = dict(color='C1', lw=2, ls='--')

    # Moments for true
    mu_true = true.mean(axis=0)                   # E[X_true]
    var_true = true.var(axis=0, ddof=0)           # Var[X_true]
    sigma_true = np.sqrt(var_true)                # std-dev band (±1σ)

    # Moments for pred
    mu_pred = pred.mean(axis=0)                   # E[X_pred]
    var_pred = pred.var(axis=0, ddof=0)           # Var[X_pred]
    sigma_pred = np.sqrt(var_pred)                # std-dev band (±1σ)

    fig, ax = plt.subplots(figsize=figsize)

    if show_samples:
        # Light sample trajectories for context
        ax.plot(t, true.T, color='C0', alpha=0.15, lw=0.5)
        ax.plot(t, pred.T, color='C1', alpha=0.15, lw=0.5)

    # True variance band
    ax.fill_between(
        t,
        mu_true - sigma_true,
        mu_true + sigma_true,
        **true_var_style,
        label='true ±1 std. dev.'
    )

    # Pred variance band
    ax.fill_between(
        t,
        mu_pred - sigma_pred,
        mu_pred + sigma_pred,
        **pred_var_style,
        label='pred ±1 std. dev.'
    )

    # Mean lines
    ax.plot(t, mu_true, **true_mean_style, label='true mean')
    ax.plot(t, mu_pred, **pred_mean_style, label='pred mean')

    ax.set_xlabel('time')
    ax.set_ylabel('value')
    ax.legend()
    fig.tight_layout()

    if save_to is not None:
        fig.savefig(save_to)
        plt.cla()
        plt.clf()
        return

    if show:
        plt.show()
    else:
        return ax


def vector_field_movie(
    vecs,
    c="k",
    n_samples=None,
    xlim=None,
    ylim=None,
    alpha=1,
    frames=60,
    t=None,
    title="",
    interval=100,
    save_to=None,
    show=True,
    fps=10,
    scale=None,
):
    """
    Animate a 2D vector field that evolves in time.

    Parameters
    ----------
    vecs : array-like, shape (T, N, 2)
        Time series of vectors (u, v) for N points over T timesteps.
        The spatial positions of the vectors are fixed and generated
        automatically on a regular grid.
    c : color or sequence of colors
        Color(s) passed to `ax.quiver`.
    n_samples : int or None
        If not None, randomly subsample N to this many vectors.
    xlim, ylim : [min, max] or None
        Axis limits. If None, they are inferred from the generated grid.
    alpha : float
        Arrow transparency.
    frames : int
        Max number of frames to show in the animation (subsamples in time).
    t : array-like or None
        Time stamps, length T. If None, uses np.arange(T).
    title : str
        Prefix for the title; time is appended as `t=...`.
    interval : int
        Delay between frames in milliseconds.
    save_to : str or Path or None
        If not None, save as GIF to this path (with .gif suffix).
    show : bool
        If True, return HTML for inline display (Jupyter).
    fps : int
        Frames per second when saving as GIF.
    scale : float or None
        Passed to `ax.quiver(scale=...)` to control arrow scaling.
    """

    vecs = np.asarray(vecs)
    assert vecs.ndim == 3 and vecs.shape[-1] == 2, "vecs must be (T, N, 2)"

    # Optional subsampling over N
    T, N, _ = vecs.shape
    if n_samples is not None and n_samples < N:
        idx = np.random.choice(N, size=n_samples, replace=False)
        idx = np.asarray(idx, dtype=np.int32)
        vecs = vecs[:, idx, :]
        N = n_samples

    # Generate a fixed 2D grid of positions for the vectors
    side = int(np.ceil(np.sqrt(N)))
    xs, ys = np.meshgrid(np.arange(side), np.arange(side))
    X = xs.ravel()[:N].astype(float)
    Y = ys.ravel()[:N].astype(float)

    # Axis limits
    if xlim is None:
        xlim = [X.min() - 0.5, X.max() + 0.5]
    if ylim is None:
        ylim = [Y.min() - 0.5, Y.max() + 0.5]

    fig, ax = plt.subplots()
    q = ax.quiver(
        X,
        Y,
        vecs[0, :, 0],
        vecs[0, :, 1],
        alpha=alpha,
        color=c,
        scale_units="xy",
        scale=scale,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", "box")

    tx = ax.set_title("Frame 0")

    # Time handling + temporal subsampling
    time = T
    if t is None:
        t = np.arange(time)
    t = np.asarray(t)
    assert len(t) == time, "t must have same length as time dimension of vecs"

    inc = max(time // frames, 1)
    t_frames = t[::inc]
    vecs_frames = vecs[::inc]

    frames_data = list(zip(vecs_frames, t_frames, strict=False))

    def animate(frame):
        v, t_val = frame  # v: (N, 2)
        q.set_UVC(v[:, 0], v[:, 1])
        tx.set_text(f"{title} t={t_val:.2f}")
        return q, tx

    ani = FuncAnimation(
        fig,
        animate,
        frames=frames_data,
        interval=interval,
        blit=False,
    )
    plt.close(fig)

    if save_to is not None:
        p = Path(save_to).with_suffix(".gif")
        ani.save(p, writer="pillow", fps=fps)

    if show:
        return HTML(ani.to_jshtml())

from pathlib import Path

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pgf import FigureCanvasPgf

PLOT_PATH = Path('./plots/')


def reset_style():
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def set_style(latex=True):
    # reset to default
    reset_style()
    base = sns.color_palette("Set1", 9)
    order = [1, 4, 3, 0, 2] + list(range(5, 9))
    new_pal = [base[i] for i in order]

    sns.set_palette(new_pal)

    # style
    matplotlib.style.use('seaborn-v0_8-whitegrid')
    matplotlib.rcParams["legend.frameon"] = True
    matplotlib.rcParams["lines.linewidth"] = 1.4
    matplotlib.rcParams["axes.linewidth"] = 0.8
    matplotlib.rcParams["axes.edgecolor"] = 'black'
    matplotlib.rcParams["ytick.major.size"] = 2

    mpl.rcParams["image.interpolation"] = "lanczos"   # keeps imshow sharp
    # fallback if you forget dpi=
    mpl.rcParams["savefig.dpi"] = 200

    mpl.rcParams.update({
        "text.usetex": True,               # route all text through LaTeX
        "font.family": "serif",            # ask for a serif font

        "font.serif": ["Computer Modern Roman"],  # article default
        # article-size font palette (10 pt base):
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })

    # Globally adjust layout to minimize whitespace:
    mpl.rcParams['figure.autolayout'] = True
    mpl.rcParams['savefig.pad_inches'] = 0
    mpl.rcParams['savefig.bbox'] = 'tight'

    if latex:
        matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'pgf.rcfonts': False,

        })

    # plt.gcf().set_tight_layout(True)
    # matplotlib.rcParams["font.size"] = 19
    # matplotlib.rcParams["legend.fontsize"] = 19
    # matplotlib.rcParams["ytick.labelsize"] = 19
    # matplotlib.rcParams["xtick.labelsize"] = 19
    # matplotlib.rcParams["axes.labelsize"] = 27


def save_show(name='', path=PLOT_PATH, save=True, show=True, format='pgf'):

    if save:
        plt.savefig(path/name, format=format,
                    bbox_inches='tight', pad_inches=0.01)
    if show:
        plt.show()


def get_colors():
    return plt.rcParams['axes.prop_cycle'].by_key()['color']


import math
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def arr_if_scalar(e):
    if isinstance(e, Iterable) and type(e) is not str:
        return e
    else:
        return [e]


def factor_int_close_to_square(n):
    d = math.ceil(math.sqrt(n))
    opt = math.inf
    opt_o = (0, 0)
    off = [(0, 0), (0, -1), (-1, 1), (-2, 1), (-2, 0)]
    for (l, r) in off:
        extra = ((d+l)*(d+r)) - n
        if extra < opt and extra >= 0:
            opt = extra
            opt_o = (l, r)
    ans = [d+opt_o[0], d+opt_o[1]]
    ans.sort()
    return tuple(ans)


def group_by(dataframe, cols):

    dfs, labels = [], []
    if len(cols) > 0 and cols is not None:
        if len(cols) == 1:
            grouped = dataframe.groupby(cols[0], dropna=False)
        else:
            grouped = dataframe.groupby(cols, dropna=False)

        for (vals, group) in grouped:
            vals = arr_if_scalar(vals)

            # build label by group vals
            label = ', '.join([f'{k}={v}' for k, v in zip(cols, vals)])
            dfs.append(group)
            labels.append(label)

    else:
        dfs.append(dataframe)
        labels.append('')

    return dfs, labels


def get_hw_from_cols(df, cols):
    # makes first title col vary by row
    d1, d2 = 1, 1
    if len(cols) == 1:
        n = len(pd.unique(df[cols[0]]))
        return factor_int_close_to_square(n)

    for i, t in enumerate(cols):
        n = len(pd.unique(df[t]))
        if i == 0:
            d1 *= n
        else:
            d2 *= n
    return d1, d2


def subplt_arr(d1, d2=None, **kwargs):
    if d2 is None:
        d1, d2 = factor_int_close_to_square(d1)
    fig, axarr = plt.subplots(d1, d2, **kwargs)

    return fig, flatten(axarr)


def set_hw(hws, fig, size):
    H, W = 1, 1
    for (h, w, _) in hws:
        H, W = H*h, W*w
    W = size[0]*W
    H = size[1]*H + 1.35*H  # extra for title room
    fig.set_size_inches(W, H)
    return H, W


def flatten(a):
    return a.flat if hasattr(a, 'flat') else [a]


def plot_df_nested(df, plotter, layers=[[]], size=(8, 6), fig_i=0, pass_fig=False,
                   sharex=True, sharey=True, color=False, stroke=True, title=None, colors=None, show=True,
                   layout={'h_pad': 0.2, 'w_pad': 0.2, 'margin': 1}):

    # preprocess so always arr of arrs
    layers = [arr_if_scalar(l) for l in layers]

    # globals
    N_nest = len(layers)
    weights = ['black', 'bold', 'medium', 'normal', 'light']
    sizes = ['xx-large', 'x-large', 'large', 'medium', 'small']
    hws = set()
    all_figs = []
    if colors is None:
        colors = [str(i) for i in np.linspace(0.86, 1.0, N_nest+1)]
    layers.append(None)  # add dummy

    if title is None:
        layers_str = ['-'.join(l) for l in layers[:-1]]
        title = ' X '.join(layers_str)

    # kinds indicates fig, subfig, or ax
    kinds = ['sub'] * N_nest
    for i in range(fig_i+1):
        kinds[i] = 'fig'
    kinds.append('ax')

    def build_fig_group(df, root, label, cols, depth, hws):

        # configure self
        cur_kind = kinds[depth]

        # ax
        if cur_kind == 'ax':
            if pass_fig:
                root.suptitle(label)
            else:
                root.set_title(label)
            plotter(df, root)
            return hws

        # sub
        if cur_kind == 'sub':
            root.suptitle(
                f'{label}', fontsize=sizes[depth], fontweight=weights[depth])
            if color:
                root.set_facecolor(colors[depth])
            if stroke:
                root.set_linewidth(1)
                root.set_edgecolor('black')
        # fig
        if cur_kind == 'fig':
            if color:
                root.set_facecolor(colors[depth])
            root.suptitle(
                label, fontsize=sizes[depth], fontweight=weights[depth])
            all_figs.append(root)
            hws = set()  # if we made a new fig reset hw

        # build subgrid
        # get sub grid dims
        h, w = get_hw_from_cols(df, cols)
        dfs, labels = group_by(df, cols)
        hws.add((h, w, depth))
        next_kind = kinds[depth+1]

        if next_kind == 'fig':
            grid = []
            for i in range(len(dfs)):
                grid.append(plt.figure(constrained_layout=True))
        elif next_kind == 'ax':
            if pass_fig:
                grid = flatten(root.subfigures(h, w))
            else:
                grid = flatten(root.subplots(
                    h, w, sharex=sharex, sharey=sharey))
        elif next_kind == 'sub':
            grid = flatten(root.subfigures(h, w))

        # recurse
        for i in range(len(dfs)):
            hws = build_fig_group(
                dfs[i], grid[i], labels[i], layers[depth+1], depth+1, hws)

        return hws

    depth = 0
    root = plt.figure(constrained_layout=True)

    # run recursion
    hws = build_fig_group(df, root, title, layers[depth], depth, set())

    # set height and width of figs
    H, W = 1, 1
    for (h, w, _) in hws:
        H, W = H*h, W*w
    W = size[0]*W
    H = size[1]*H + 1.35*H  # extra for title room
    margin = layout['margin']
    for fig in all_figs:
        fig.set_size_inches(W, H)
        fig.set_constrained_layout_pads(
            w_pad=layout['w_pad'], h_pad=layout['h_pad'], hspace=margin/H, wspace=margin/W)

    if show:
        plt.show()
    else:
        return root


def series_plotter(y, x=None, lines=[], legend=True, logy=False):

    y_strs = arr_if_scalar(y)
    x_strs = arr_if_scalar(x)

    for i in range(len(y_strs) - len(x_strs)):
        x_strs.append(x_strs[0])

    def plotter(df, ax):

        dfs, labels = group_by(df, lines)

        for df, label in zip(dfs, labels):

            for x_str, y_str in zip(x_strs, y_strs):

                y_vals = df[y_str].to_numpy()

                if x_str is None:
                    x = np.arange(0, len(y_vals[0]))
                else:
                    x = df[x_str].to_numpy()[0]

                y, e = np.mean(y_vals, axis=0), np.std(y_vals, axis=0)

                if logy:
                    ax.semilogy(x, y, '.-', label=f'{label} [{y_str}]')
                else:
                    ax.plot(x, y, '.-', label=f'{label} [{y_str}]')
                    ax.fill_between(x, y-e, y+e, alpha=0.20)

        if legend:
            ax.legend()

    return plotter


def flatten_cols(df, cols):
    recs = df.to_dict('records')
    new_records = []
    for r in recs:
        serieses = [r[c] for c in cols]
        for pts in zip(*serieses):
            new_d = {}
            for k, pt in zip(cols, pts):
                new_d[k] = pt
            new_r = {**r, **new_d}
            new_records.append(new_r)

    df = pd.DataFrame(new_records)
    return df


def sns_plotter(**kwargs):

    def plotter(df, ax):
        df = flatten_cols(df, [kwargs['x'], kwargs['y']])
        sns.lineplot(data=df, ax=ax, **kwargs)

    return plotter

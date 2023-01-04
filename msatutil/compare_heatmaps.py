from msatutil.msat_interface import *


def compare_heatmaps(
    msat_1: msat_collection,
    msat_2: msat_collection,
    labs: Annotated[Sequence[str], 2],
    var: str,
    grp: Optional[str] = None,
    sv_var: Optional[str] = None,
    option: Optional[str] = None,
    option_axis: Optional[str] = None,
    hist_nbins: int = 40,
    hist_xlim: Annotated[Sequence[float], 2] = None,
    vminmax: Optional[Annotated[Sequence[float], 2]] = None,
    ratio: bool = False,
    ylim: Annotated[Sequence[float], 2] = [25, 200],
    save_path: Optional[str] = None,
    extra_id: Optional[int] = None,
    data_in: Optional[np.ndarray] = None,
    lon_lim: Optional[Annotated[Sequence[float], 2]] = None,
    lat_lim: Optional[Annotated[Sequence[float], 2]] = None,
    res: float = 20,
    scale: float = 1.0,
    exp_fmt: bool = True,
) -> Tuple[plt.Figure, Annotated[Sequence[plt.Axes], 3], np.ndarray, np.ndarray]:
    """
    Make a 3-panel plot comparing a given variables between two sets of msat files by showing one heatmap for each and one histogram
    msat_1: first msat_interface.msat_collection object
    msat_2: second msat_interface.msat_collection object
    labs: list of 2 labels to use for the legend and subplots titles for msat_1 and msat_2, respectively
    var: partial variable name (will use msat_nc.fetch method to get it)
    grp: if givem use msat_nc.get_var instead of msat_nc.fetch and var must be the exact variable name
    sv_var: when the variable is one of APosterioriState or APrioriState, this selects for the state vector variable
    option: can be used to get stats from a 3d variable (any numpy method e.g. 'max' 'nanmax' 'std')
    option_axis: the axis along which the stat is applied
    hist_nbins: number of bins for the histogram 
    hist_xlim: horizontal axis range for the histogram
    vminmax: [min,max] of the heatmap colorbars
    ratio: if True, divide the variable by its median
    ylim: vertical axis range for the heatmaps
    save_path: full path to save the figure
    extra_id: when using 3D data, slice the 3rd dimension with this index
    data_in: list of the data to plot [x1,x2] corresponding to msat_1 and msat_2, if given, uses this data instead of reading the variable with pmesh_prep
    lon_lim: [min,max] longitudes for the gridding
    lat_lim: [min,max] latitudes for the gridding
    res: the resolution (in meters) of the grid with lon_lim and lat_lim are given
    scale: quantity to multiply the variable with (can be useful to avoid overflow in the standard deviation of column amounts)
    exp_fmt: if True, use .3e format for stats in the histogram legend. If false use .2f format
    """

    fig, ax = plt.subplot_mosaic(
        [["upper left", "right"], ["lower left", "right"]], gridspec_kw={"width_ratios": [2.5, 2]},
    )
    fig.set_size_inches(12, 10)
    plt.subplots_adjust(wspace=0.3)

    gridded = (lon_lim is not None) and (lat_lim is not None)

    ax["upper left"].set_title(labs[0])
    ax["lower left"].set_title(labs[1])
    if gridded:
        for i in ["upper left", "lower left"]:
            ax[i].set_ylim(lat_lim)
            ax[i].set_xlim(lon_lim)
    else:
        ax["upper left"].set_ylim(ylim)
        ax["lower left"].set_ylim(ylim)

    for curax in [ax["upper left"], ax["lower left"]]:
        if gridded:
            curax.set_ylabel("Latitude")
        else:
            curax.set_ylabel("cross-track index")
    if gridded:
        ax["lower left"].set_xlabel("Longitude")
    else:
        ax["lower left"].set_xlabel("along-track index")
    fig.suptitle(
        f"{datetime.strftime(msat_1.dates[0],'%Y%m%dT%H%M%S')} to {datetime.strftime(msat_1.dates[-1],'%Y%m%dT%H%M%S')}"
    )

    # make the heatmaps
    if data_in:
        x1, x2 = data_in
    else:
        if gridded:
            lon_grid1, lat_grid1, x1 = msat_1.grid_prep(
                var,
                grp=grp,
                sv_var=sv_var,
                extra_id=extra_id,
                option=option,
                option_axis=option_axis,
                lon_lim=lon_lim,
                lat_lim=lat_lim,
                res=res,
            )
            lon_grid2, lat_grid2, x2 = msat_2.grid_prep(
                var,
                grp=grp,
                sv_var=sv_var,
                extra_id=extra_id,
                option=option,
                option_axis=option_axis,
                lon_lim=lon_lim,
                lat_lim=lat_lim,
                res=res,
            )
        else:
            x1 = msat_1.pmesh_prep(
                var,
                grp=grp,
                sv_var=sv_var,
                extra_id=extra_id,
                option=option,
                option_axis=option_axis,
            )
            x2 = msat_2.pmesh_prep(
                var,
                grp=grp,
                sv_var=sv_var,
                extra_id=extra_id,
                option=option,
                option_axis=option_axis,
            )

        x1 = x1 * scale
        x2 = x2 * scale

        if ratio:
            x1 = x1 / np.nanmedian(x1)
            x2 = x2 / np.nanmedian(x2)
        if msat_1.use_dask:
            x1 = x1.compute()
        if msat_2.use_dask:
            x2 = x2.compute()

    if vminmax is None:
        vmin = np.min([np.nanmin(x1), np.nanmin(x2)])
        vmax = np.max([np.nanmax(x1), np.nanmax(x2)])
        vminmax = [vmin, vmax]
    if gridded:
        m1 = ax["upper left"].pcolormesh(
            lon_grid1, lat_grid1, x1, cmap="viridis", vmin=vminmax[0], vmax=vminmax[1]
        )
        m2 = ax["lower left"].pcolormesh(
            lon_grid2, lat_grid2, x2, cmap="viridis", vmin=vminmax[0], vmax=vminmax[1]
        )
    else:
        m1 = ax["upper left"].pcolormesh(x1, cmap="viridis", vmin=vminmax[0], vmax=vminmax[1])
        m2 = ax["lower left"].pcolormesh(x2, cmap="viridis", vmin=vminmax[0], vmax=vminmax[1])
    if var == "dp":
        lab = "$\Delta P$"
    elif sv_var:
        lab = sv_var
    else:
        lab = var

    units = msat_1.fetch_units(var)
    if units:
        lab = f"{lab} ({units})"
    print("units", units)

    plt.colorbar(m1, label=lab, ax=[ax["upper left"], ax["lower left"]])

    if hist_xlim is None:
        hist_xlim = [
            np.nanmin([np.nanmin(x1), np.nanmin(x2)]),
            np.nanmax([np.nanmax(x1), np.nanmax(x2)]),
        ]

    # make the histograms
    maxval1 = make_hist(
        ax["right"],
        x1[np.isfinite(x1)].flatten(),
        labs[0],
        "blue",
        rng=hist_xlim,
        nbins=hist_nbins,
        exp_fmt=exp_fmt,
    )
    maxval2 = make_hist(
        ax["right"],
        x2[np.isfinite(x2)].flatten(),
        labs[1],
        "red",
        rng=hist_xlim,
        nbins=hist_nbins,
        exp_fmt=exp_fmt,
    )

    ax["right"].set_ylim(0, 1.25 * np.max([maxval1, maxval2]))

    if save_path:
        fig.savefig(save_path)

    return fig, ax, x1, x2


def make_hist(
    ax: plt.Axes,
    x: Sequence[float],
    label: str,
    color: str,
    rng: Optional[Annotated[Sequence[float], 2]] = None,
    nbins: Optional[int] = None,
    exp_fmt: bool = True,
):
    """
    Make a historgram of the data in x
    ax: matplotlib axes object
    x: array of data
    label: label for the legend
    color: color of the bars
    rng: range of the horizontal axis
    nbins: number of bins for the histogram
    """
    if rng is not None:
        x_rng = x[(x >= rng[0]) & (x <= rng[1])]
        x_mean = np.nanmean(x_rng)
        x_std = np.nanstd(x_rng, ddof=1)
        x_med = np.nanmedian(x_rng)
        if exp_fmt:
            label = (
                label
            ) = f"{label}\n$\mu\pm\sigma$: {x_mean:.3e}$\pm${x_std:.3e}\nmedian: {x_med:.3e} "
        else:
            label = (
                label
            ) = f"{label}\n$\mu\pm\sigma$: {x_mean:.2f}$\pm${x_std:.2f}\nmedian: {x_med:.2f} "
        bin_vals, bin_edges, patches = ax.hist(
            x,
            edgecolor=color,
            facecolor="None",
            label=label,
            range=rng,
            bins=nbins,
            histtype="step",
        )
    else:
        x_mean = np.nanmean(x)
        x_std = np.nanstd(x, ddof=1)
        x_med = np.nanmedian(x)
        if exp_fmt:
            label = f"{label}\n$\mu\pm\sigma$: {x_mean:.3e}$\pm${x_std:.3e}\nmedian: {x_med:.3e} "
        else:
            label = f"{label}\n$\mu\pm\sigma$: {x_mean:.2f}$\pm${x_std:.2f}\nmedian: {x_med:.2f} "
        bin_vals, bin_edges, patches = ax.hist(
            x, edgecolor=color, facecolor="None", label=label, histtype="step",
        )
    ax.axvline(x=x_med, color=color, linestyle="--")
    ax.legend(frameon=False)
    return np.max(bin_vals)


def main():
    parser = argparse.ArgumentParser(
        description="Make a 4 panel plots compating a given variables between two sets of msat files by showing one heatmap for each and one histogram"
    )
    parser.add_argument(
        "path1", help="full path to the directory where the first set of msat files exists",
    )
    parser.add_argument(
        "path2", help="full path to the directory where the second set of msat files exists",
    )
    parser.add_argument("var", help="variable name")
    parser.add_argument(
        "--sv-var", default=None, help="exact SubStateName of the state vector variable"
    )
    parser.add_argument(
        "--labs",
        nargs=2,
        default=["top", "bottom"],
        help="label corresponding to the data in path1 and path2, respectively",
    )
    parser.add_argument(
        "--extra-id",
        type=int,
        default=0,
        help="integer to slice a third index (e.g. along wmx_1 for Radiance_I (wmx_1,jmx,imx)) only does something for 3D variables",
    )
    parser.add_argument("--search", default="proxy.nc", help="string pattern to select msat files")
    parser.add_argument(
        "--vminmax", nargs=2, type=float, default=None, help="min and max values for the colorbar",
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=[25, 200],
        help="sets vertical axis limits when plotting heatmaps with cross/along track indices",
    )
    parser.add_argument(
        "--lon-lim",
        nargs=2,
        type=float,
        default=None,
        help="if given with --lat-lim, regrid the data to --res for the heatmaps",
    )
    parer.add_argument(
        "--res",
        type=float,
        default=20,
        help="if --lon-lim and --lat-lim are given, regrid the data to this resolution (in meters) for the heatmaps",
    )
    parser.add_argument(
        "--lat-lim",
        nargs=2,
        type=float,
        default=None,
        help="if given with --lon-lim, regrid the data to --res for the heatmaps",
    )
    parser.add_argument(
        "--hist-xlim",
        nargs=2,
        type=float,
        default=[-25, 25],
        help="sets horizontal axis limits for histograms",
    )
    parser.add_argument(
        "--hist-nbins", default=40, type=int, help="sets the number of bins for the histograms",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="quantity to multiply the variable with (can be useful to avoid overflow in the standard deviation of column amounts)",
    )
    parser.add_argument(
        "-r",
        "--ratio",
        action="store_true",
        help="if given, plots the variable divided by its median",
    )
    parser.add_argument(
        "-s", "--save-path", default="", help="full filepath to save the plot (includes filename)",
    )
    parser.add_argument(
        "--use-dask", action="store_true", help="if given, use dask to handle the data"
    )
    args = parser.parse_args()

    if args.sv_var and args.var not in ["APosterioriState", "APrioriState"]:
        raise Exception(
            'When --sv-var is given, var must be one of ["APrioriState","APosterioriState"]'
        )

    for path in [args.path1, args.path2]:
        if not os.path.isdir(path):
            raise Exception(f"{path} is not a valid path")

    msat_1 = msat_collection(
        [
            os.path.join(args.path1, i)
            for i in os.listdir(args.path1)
            if args.search in i and i.endswith("proxy.nc")
        ],
        use_dask=args.use_dask,
    )
    msat_2 = msat_collection(
        [
            os.path.join(args.path2, i)
            for i in os.listdir(args.path2)
            if args.search in i and i.endswith("proxy.nc")
        ],
        use_dask=args.use_dask,
    )

    fig, ax, x1, x2 = compare_heatmaps(
        msat_1,
        msat_2,
        args.labs,
        args.var,
        sv_var=args.sv_var,
        hist_nbins=args.hist_nbins,
        hist_xlim=args.hist_xlim,
        vminmax=args.vminmax,
        ratio=args.ratio,
        ylim=args.ylim,
        save_path=args.save_path,
        extra_id=args.extra_id,
        lon_lim=args.lon_lim,
        lat_lim=args.lat_lim,
        res=args.res,
        scale=args.scale,
    )

    return fig, ax, x1, x2


if __name__ == "__main__":
    plt.ion()
    fig, ax, x1, x2 = main()

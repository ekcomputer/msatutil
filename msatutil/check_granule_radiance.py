import os
import argparse
from pylab import *
from typing import Optional
from netCDF4 import Dataset
import warnings


def check_granule_radiances(
    l1_infile: str, threshold: float = 0.04, save_path: Optional[str] = None
) -> None:
    """
    Compute and plot the standard deviation of normalized radiances in the granule at l1_infile
    Also plot the radiances of the spectra that fall below a given threshold and include the pixel index in the legend

    l1_infile: full path to MethaneSAT/AIR L1B file
    threshold: value of the standard deviation of normalized radiances below which spectra are considered "bad"
    save_path: full path to the output directory where figures will be saved
    """

    with Dataset(l1_infile, "r") as l1:
        radiance = l1["Band1/Radiance"][:]
        if "w1" in l1.dimensions:
            nalong = l1.dimensions["y"].size
            radiance = radiance.transpose(1, 2, 0)
        else:
            nalong = l1.dimensions["along_track"].size

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rad_across_mean = np.nanmean(
            radiance, axis=(0, 2)
        )  # squish the radiance to get an array with across_track size
        valid_across_ids = np.where(~np.isnan(rad_across_mean))[
            0
        ]  # get the valid across track indices
        std_norm_rad = np.nanstd(radiance, axis=2, ddof=1) / np.nanmax(radiance, axis=2)

    bad_ids = np.where(std_norm_rad < threshold)

    flat_std_norm_rad = std_norm_rad.flatten()

    plot(flat_std_norm_rad, marker="o", linewidth=0)
    title("Standard deviation of normalized radiances")
    xlabel("pixel index")
    if save_path is not None:
        save_name = os.path.join(
            save_path, os.path.basename(l1_infile).replace(".nc", "_std_norm_rad.png")
        )
        gcf().savefig(save_name)
    else:
        show()
    clf()
    if len(bad_ids):
        for i, j in zip(bad_ids[0], bad_ids[1]):
            plot(radiance[i, j], label=f"along_index: {i}; across_index: {j}")
        legend()
        xlabel("spectral index")
        ylabel("Radiance")
        if save_path is not None:
            save_name = os.path.join(
                save_path,
                os.path.basename(l1_infile).replace(".nc", "_bad_spectra.png"),
            )
            gcf().savefig(save_name)
        else:
            show()


def main():
    parser = argparse.ArgumentParser(
        description="Compute and plot the standard deviation of normalized radiances in the granule at l1_infile."
        " Also plot the radiances of the spectra that fall below a given threshold and include the pixel index in the legend"
    )
    parser.add_argument("l1_infile")
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.04,
        help="threshold for the identifying bad standard deviation of normalized radiances",
    )
    parser.add_argument(
        "-s",
        "--save-path",
        default=None,
        help="full path to the DIRECTORY of the output figures, does not save figures by default",
    )
    args = parser.parse_args()

    check_granule_radiances(
        args.l1_infile, threshold=args.threshold, save_path=args.save_path
    )


if __name__ == "__main__":
    main()

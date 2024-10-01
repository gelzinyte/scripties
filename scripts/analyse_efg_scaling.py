import numpy as np
from pathlib import Path
from matplotlib import gridspec

from pypdf import PdfWriter

from ase.io import read, write

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from util import efg as efgu


cwd = Path().resolve()

outdir = Path(f"/raven/u/egg/mounted/EFGs") / "0.data_distributions"
outdir.mkdir(exist_ok=True, parents=True)

dft_label = "efg"
mace_label = "mace_efg"

mace_path = "models/EFGsMACE_test.model"

train_fn_in = "/raven/u/egg/data/lto_efgs_from_ange/LTO42-ratttle-pristine-fracscale_train_1312_b73de8.xyz"

# --------------------
# plot
# --------------------


ats_train = read(train_fn_in, ":")
efgs = efgu.extract_efgs(ats_train, efg_label=dft_label)

elements = ["Li", "O", "Ti"]

ref_skip = [
    "quaternions",
    "etas",
    "phis",
    "thetas",
    "evecs",
    "tilde_Cqs",
    "tilde_etas",
    "determinants",
    "tilde_Cqs",
    "tilde_etas",
]

determinants_dict = {}
for element in elements:

    data = efgu.get_props(efgs[element], element)
    determinants_dict[element] = data["determinants"]

selected_properties = [key for key in data if key not in ref_skip]

labels = [
    "raw",
    #  x   "scaled_by_mean_det",          
    #  x   "scaled_by_std_det",        
    #  m   "scaled_by_mean_sqrt_det",      
    #  m   "scaled_by_std_sqrt_det",
    #  !   "scaled_by_mean_cubert_det",   
    #  x   "scaled_by_min_max_det",
    #  x   "scaled_by_min_max_abs_det",
    #  x   "scaled_by_mean_evals",
    #  !   "scaled_by_std_evals",
    #  v   "scaled_by_std_evals_shifted_by_mean",
    # "scaled by std_sqrt_det_shifted_by_mean",
    #  x   "scaled_by_mean_cubert_signed_det",
    #  !   "scaled_by_mean_abs_evals",
    "scaled_by_std_abs_evals",
]

num_rows = len(selected_properties) + 8
num_columns = len(elements)
side = 4
figsize = (num_columns * side, num_rows * side)

for prefix in labels:

    pdf_fn = outdir / f"{prefix}.pdf"
    if pdf_fn.exists():
        continue

    axes_dict = {}

    fig = plt.figure(figsize=figsize, constrained_layout=False)
    grid = fig.add_gridspec(num_rows, num_columns, wspace=0.4, hspace=0.4)

    pdfs_out_fn = outdir / f"{prefix}.pdf"

    for el_idx, element in enumerate(elements):

        if prefix == "raw":
            efgs_scaled_shifted = efgs[element]
        elif prefix == "scaled_by_mean_det":
            mean_det = np.mean(determinants_dict[element])
            func = lambda x: x / mean_det
            efgs_scaled_shifted = np.apply_along_axis(
                func1d=func, axis=0, arr=efgs[element]
            )
        elif prefix == "scaled_by_std_det":
            std_det = np.std(determinants_dict[element])
            func = lambda x: x / std_det
            efgs_scaled_shifted = np.apply_along_axis(
                func1d=func, axis=0, arr=efgs[element]
            )
        elif prefix == "scaled_by_mean_sqrt_det":
            scale = np.mean(np.sqrt(np.abs(determinants_dict[element])))
            func = lambda x: x / scale
            efgs_scaled_shifted = np.apply_along_axis(
                func1d=func, axis=0, arr=efgs[element]
            )
        elif prefix == "scaled_by_std_sqrt_det":
            scale = np.std(np.sqrt(np.abs(determinants_dict[element])))
            func = lambda x: x / scale
            efgs_scaled_shifted = np.apply_along_axis(
                func1d=func, axis=0, arr=efgs[element]
            )
        elif prefix == "scaled_by_mean_cubert_det":
            scale = np.mean(np.cbrt(np.abs(determinants_dict[element])))
            func = lambda x: x / scale
            efgs_scaled_shifted = np.apply_along_axis(
                func1d=func, axis=0, arr=efgs[element]
            )
        elif prefix == "scaled_by_min_max_det":
            min_det = np.min(determinants_dict[element])
            max_det = np.max(determinants_dict[element])
            scale = max_det - min_det
            func = lambda x: x / scale
            efgs_scaled_shifted = np.apply_along_axis(
                func1d=func, axis=0, arr=efgs[element]
            )
        elif prefix == "scaled_by_min_max_abs_det":
            min_det = np.min(np.abs(determinants_dict[element]))
            max_det = np.max(np.abs(determinants_dict[element]))
            scale = max_det - min_det
            func = lambda x: x / scale
            efgs_scaled_shifted = np.apply_along_axis(
                func1d=func, axis=0, arr=efgs[element]
            )
        elif prefix == "scaled_by_mean_evals":
            data = efgu.get_props(efgs[element], element)
            evals = np.concatenate([data["evals_0"], data["evals_1"], data["evals_2"]])
            scale = np.mean(evals)
            func = lambda x: x / scale
            efgs_scaled_shifted = np.apply_along_axis(
                func1d=func, axis=0, arr=efgs[element]
            )
        elif prefix == "scaled_by_std_evals":
            data = efgu.get_props(efgs[element], element)
            evals = np.concatenate([data["evals_0"], data["evals_1"], data["evals_2"]])
            scale = np.std(evals)
            func = lambda x: x / scale
            efgs_scaled_shifted = np.apply_along_axis(
                func1d=func, axis=0, arr=efgs[element]
            )
        elif prefix == "scaled_by_std_evals_shifted_by_mean":
            data = efgu.get_props(efgs[element], element)
            evals = np.concatenate([data["evals_0"], data["evals_1"], data["evals_2"]])
            scale = np.std(evals)
            shift = np.mean(evals)
            func = lambda x: (x - shift) / scale
            efgs_scaled_shifted = np.apply_along_axis(
                func1d=func, axis=0, arr=efgs[element]
            )
        elif prefix == "scaled_by_std_cbrt_det_shifted_by_mean":
            cbrt_dets = np.cbrt(np.abs(determinants_dict[element]))
            scale = np.std(cbrt_dets)
            shift = np.mean(cbrt_dets)
            func = lambda x: (x - shift) / scale
            efgs_scaled_shifted = np.apply_along_axis(
                func1d=func, axis=0, arr=efgs[element]
            )
        elif prefix == "scaled_by_mean_cubert_signed_det":
            scale = np.mean(np.cbrt(determinants_dict[element]))
            func = lambda x: x / scale
            efgs_scaled_shifted = np.apply_along_axis(
                func1d=func, axis=0, arr=efgs[element]
            )
        elif prefix == "scaled_by_mean_abs_evals":
            data = efgu.get_props(efgs[element], element)
            evals = np.concatenate([data["evals_0"], data["evals_1"], data["evals_2"]])
            scale = np.mean(np.abs(evals))
            func = lambda x: x / scale
            efgs_scaled_shifted = np.apply_along_axis(
                func1d=func, axis=0, arr=efgs[element]
            )
        elif prefix == "scaled_by_std_abs_evals":
            data = efgu.get_props(efgs[element], element)
            evals = np.concatenate([data["evals_0"], data["evals_1"], data["evals_2"]])
            scale = np.std(np.abs(evals))
            func = lambda x: x / scale
            efgs_scaled_shifted = np.apply_along_axis(
                func1d=func, axis=0, arr=efgs[element]
            )



        data = efgu.get_props(efgs_scaled_shifted, element)

        for prop_idx, prop in enumerate(selected_properties):

            if "eval" in prop:
                color="tab:orange"
            elif "L2_ir" in prop:
                color="tab:green"
            else:
                color="tab:blue"

            ax = fig.add_subplot(grid[prop_idx, el_idx])
            
            if prop_idx not in axes_dict:
                axes_dict[prop_idx] = []
            axes_dict[prop_idx].append(ax)

            dft_vals = data[prop]

            ref_multival = ["efg", "efg_L2_ir"]
            ref_multival += [f"{prefix}{label}" for label in ref_multival]
            if prop in ref_multival:
                entries = np.arange(5)
            else:
                entries = np.arange(1)

            for entry in entries:

                if len(entries) == 1:
                    ax.hist(dft_vals, bins=20, label=f"dft_train", color=color) 

                else:
                    to_plot_train = [val[entry] for val in dft_vals]
                    ax.hist(
                        to_plot_train, bins=20, label=f"mean_det", color=color
                    )  

                ax.set_xlabel(f"DFT {prop}")
                ax.set_ylabel("Counts")
                ax.legend()

                ax.set_title(f"{element}_{prop} {entry}")

    for _, axes_set in axes_dict.items():
    
        min_vals = np.min([ax.get_xlim()[0] for ax in axes_set])
        max_vals = np.max([ax.get_xlim()[1] for ax in axes_set])
        for ax in axes_set:
            ax.set_xlim(left=min_vals, right=max_vals)


    
    plt.savefig(pdf_fn, bbox_inches="tight")
    plt.close()

    # scale by mean of determinants
    # scale by mean of sqrt(determinants)
    # scale by mean of cubicrt(determinants)
    # shift by mean of determinants scale by ???
    # scale by min max determinants
    # scale by min max sqrt(determinants)

    # shift scale by det mean std
    # mean_det = np.mean(data["determinants"])
    # std_det = np.std(data["determinants"])

    # prefix = "shift_scale_by_sqrtdet_mean_std"
    # mean_det = np.mean(np.sqrt(np.abs(data["determinants"])))
    # std_det = np.std(np.sqrt(np.abs(data["determinants"])))

    # print(f"mean {element} determinant: {mean_det}, std{std_det}")

    # normalized_efgs = np.array([(val - mean_det)/std_det for val in efgs[element]])
    # data_scaled_shifted = efgu.get_props(normalized_efgs, element)

    # prefix="normalized_"
    # normalized_efgs = np.array([val / (np.sqrt(np.abs(mean_det))) for val in efgs[element]])
    # data_scaled_shifted = efgu.get_props(normalized_efgs, element)

    # data = efgu.get_props(efgs[element], element)
    # min_eval = np.min(data["evals"])
    # max_eval = np.max(data["evals"])

    # prefix = "min_max_eigenvalues_"
    # normalized_efgs = np.array([(val - min_eval)/(max_eval - min_eval) for val in efgs[element]])

    # prefix = "scale_only_min_max_eigenvalues_"
    # normalized_efgs = np.array([val/(max_eval - min_eval) for val in efgs[element]])

    # prefix = "scale_only_det_std_"
    # mean_det = np.mean(data["determinants"])
    # std_det = np.std(data["determinants"])
    # normalized_efgs = np.array([val/std_det for val in efgs[element]])

    # prefix = "scale_only_sqrt_det_std_"
    # mean_det = np.mean(np.sqrt(np.abs(data["determinants"])))
    # std_det = np.std(np.sqrt(np.abs(data["determinants"])))
    # normalized_efgs = np.array([val/std_det for val in efgs[element]])

    # mean_det = np.mean(np.sqrt(np.abs(data["determinants"])))
    # std_det = np.std(np.sqrt(np.abs(data["determinants"])))

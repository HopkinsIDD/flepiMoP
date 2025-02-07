import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import seaborn as sns
import tqdm

from .model_info import ModelInfo


def find_walkers_to_sample(inferpar, sampler_output, nsamples, nwalker, nthin):
    # Find the good walkers
    if nsamples < nwalker:
        pass
        # easy case: sample the best llik (could also do random)

    last_llik = sampler_output.get_log_prob()[-1, :]
    sampled_slots = last_llik > (last_llik.mean() - 1 * last_llik.std())
    print(
        f"there are {sampled_slots.sum()}/{len(sampled_slots)} good walkers... keeping these"
    )
    # TODO this function give back

    good_samples = sampler.get_chain()[:, sampled_slots, :]

    step_number = -1
    exported_samples = np.empty((nsamples, inferpar.get_dim()))
    for i in range(nsamples):
        exported_samples[i, :] = good_samples[
            step_number - thin * (i // (sampled_slots.sum())), i % (sampled_slots.sum()), :
        ]  # parentesis around i//(sampled_slots.sum() are very important


def plot_chains(
    inferpar, chains, llik, save_to, sampled_slots=None, param_gt=None, llik_gt=None
):
    """
    Plot the chains of the inference
    :param inferpar: the inference parameter object
    :param chains: the chains from the inference, shape (niter, nwalkers, nparam)
    :param llik: the log likelihood of the chains, shape (niter, nwalkers)
    :param save_to: the path to save the pdf
    :param sampled_slots: the slots to sample, if None, all are sampled
    :param param_gt: the ground truth parameters, shape (nparam)
    :param llik_gt: the ground truth log likelihood, shape (1)
    """
    # we plot first from the start, then the last 3/4

    niters = chains.shape[0]
    nwalkers = chains.shape[1]
    first_thresh = 0
    second_thresh = 3 * niters // 4

    if sampled_slots is None:
        sampled_slots = np.array([True] * nwalkers)

    labels = list(zip(inferpar.pnames, inferpar.subpops))

    def plot_single_chain(frompt, ax, chain, label, gt=None):
        x_plt = np.arange(frompt, niters)
        ax.plot(
            x_plt,
            chain[frompt:, sampled_slots],
            "navy",
            alpha=0.4,
            lw=1,
            label="good walkers",
        )
        ax.plot(
            x_plt,
            chain[frompt:, ~sampled_slots],
            "tomato",
            alpha=0.4,
            lw=1,
            label="bad walkers",
        )
        if gt is not None:
            ax.plot(x_plt, np.repeat(gt, len(x_plt)), "black", alpha=1, lw=2, ls="-.")
        ax.set_title(label)
        # ax.yaxis.set_label_coords(-0.1, 0.5)
        sns.despine(ax=ax, trim=False)

    print("generating chain plot")
    with PdfPages(f"{save_to}") as pdf:
        d = pdf.infodict()
        d["Title"] = "FlepiMoP Inference Chains"
        d["Author"] = "FlepiMoP Inference"
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        plot_single_chain(first_thresh, axes[0], llik, label="llik", gt=llik_gt)
        plot_single_chain(second_thresh, axes[1], llik, label="llik", gt=llik_gt)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        for sp in tqdm.tqdm(set(inferpar.subpops)):  # find unique supopulation
            these_pars = inferpar.get_parameters_for_subpop(sp)
            fig, axes = plt.subplots(
                max(len(these_pars), 2), 2, figsize=(6, (len(these_pars) + 1) * 2)
            )
            for idx, par_id in enumerate(these_pars):
                plot_single_chain(
                    first_thresh,
                    axes[idx, 0],
                    chains[:, :, par_id],
                    labels[par_id],
                    gt=param_gt[par_id] if param_gt is not None else None,
                )
                plot_single_chain(
                    second_thresh,
                    axes[idx, 1],
                    chains[:, :, par_id],
                    labels[par_id],
                    gt=param_gt[par_id] if param_gt is not None else None,
                )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def plot_fit(
    modinf: ModelInfo,
    loss,
    list_of_df,
    save_to,
    apply_transforms=True,
    plot_projections=False,
):
    with PdfPages(f"{save_to}") as pdf:
        d = pdf.infodict()
        d["Title"] = "FlepiMoP Inference Fit"
        d["Author"] = "FlepiMoP Inference"

        for j, subpop in enumerate(modinf.subpop_struct.subpop_names):
            fig, axes = plt.subplots(
                1, len(loss.statistics), figsize=(3 * len(loss.statistics), 4), sharex=True
            )

            # only one outcome
            if len(loss.statistics) == 1:
                axes = [
                    axes,
                ]

            gt_s = loss.gt[loss.gt["subpop"] == subpop].sort_index()
            first_date = max(gt_s.index.min(), list_of_df[0].index.min())
            last_date = min(gt_s.index.max(), list_of_df[0].index.max())
            gt_s = gt_s.loc[first_date:last_date].drop(["subpop"], axis=1)

            for i, (stat_name, stat) in enumerate(loss.statistics.items()):
                ax = axes[i]
                for model_df in list_of_df:
                    model_df_s = model_df[model_df["subpop"] == subpop].drop(
                        ["subpop"], axis=1
                    )  # todo sub subpop here
                    if not plot_projections:
                        model_df_s = model_df_s.loc[first_date:last_date]
                    if apply_transforms:
                        df_p = stat.apply_transforms(model_df_s[stat.sim_var].to_xarray())
                    else:
                        df_p = model_df_s[stat.sim_var].to_xarray()
                    ax.plot(df_p.date, df_p, lw=0.9, alpha=0.5, marker=".", markersize=1)
                # if True:
                #        init_df_s = outcomes_df_ref[model_df["subpop"]==subpop].drop(["subpop","time"],axis=1).loc[min(gt_s.index):max(gt_s.index)].resample("W-SAT").sum() # todo sub subpop here
                if apply_transforms:
                    gt_p = stat.apply_transforms(gt_s[stat.data_var].to_xarray())
                else:
                    gt_p = gt_s[stat.data_var].to_xarray()
                ax.plot(gt_p.date, gt_p, color="k", marker=".", lw=1)

                ax.set_title(f"{stat_name}, {subpop}")
                ax.grid(True, lw=0.5, ls="--")
                ax.set_ylim(bottom=0)
            fig.autofmt_xdate()
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

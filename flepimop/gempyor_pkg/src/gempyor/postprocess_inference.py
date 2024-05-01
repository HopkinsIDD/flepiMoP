def find_walkers_to_sample(inferpar, sampler_output, nsamples, nwalker, nthin):
    # Find the good walkers
    if nsamples < nwalker:
        pass
        # easy case: sample the best llik (could also do random)

    last_llik = sampler_output.get_log_prob()[-1, :]
    sampled_slots = last_llik > (last_llik.mean() - 1 * last_llik.std())
    print(f"there are {sampled_slots.sum()}/{len(sampled_slots)} good walkers... keeping these")
    # TODO this function give back

    good_samples = sampler.get_chain()[:, sampled_slots, :]

    step_number = -1
    exported_samples = np.empty((nsamples, inferpar.get_dim()))
    for i in range(nsamples):
        exported_samples[i, :] = good_samples[
            step_number - thin * (i // (sampled_slots.sum())), i % (sampled_slots.sum()), :
        ]  # parentesis around i//(sampled_slots.sum() are very important


def plot_chains(inferpar, sampler_output, sampled_slots=None, save_to=None):
    fig, axes = plt.subplots(inferpar.get_dim() + 1, 2, figsize=(15, (inferpar.get_dim() + 1) * 2))

    labels = list(zip(inferpar.pnames, inferpar.subpops))
    samples = sampler_output.get_chain()
    p_gt = np.load("parameter_ground_truth.npy")
    if sampled_slots is None:
        sampled_slots = [True] * inferpar.get_dim()

    import seaborn as sns

    def plot_chain(frompt, axes):
        ax = axes[0]

        ax.plot(
            np.arange(frompt, frompt + sampler_output.get_log_prob()[frompt:].shape[0]),
            sampler_output.get_log_prob()[frompt:, sampled_slots],
            "navy",
            alpha=0.2,
            lw=1,
            label="good walkers",
        )
        ax.plot(
            np.arange(frompt, frompt + sampler_output.get_log_prob()[frompt:].shape[0]),
            sampler_output.get_log_prob()[frompt:, ~sampled_slots],
            "tomato",
            alpha=0.4,
            lw=1,
            label="bad walkers",
        )
        ax.set_title("llik")
        # ax.legend()
        sns.despine(ax=ax, trim=False)
        ax.set_xlim(frompt, frompt + sampler_output.get_log_prob()[frompt:].shape[0])

        # ax.set_xlim(0, len(samples))

        for i in range(inferpar.get_dim()):
            ax = axes[i + 1]
            x_plt = np.arange(frompt, frompt + sampler_output.get_log_prob()[frompt:].shape[0])
            ax.plot(
                x_plt,
                samples[frompt:, sampled_slots, i],
                "navy",
                alpha=0.2,
                lw=1,
            )
            ax.plot(
                x_plt,
                samples[frompt:, ~sampled_slots, i],
                "tomato",
                alpha=0.4,
                lw=1,
            )
            ax.plot(x_plt, np.repeat(p_gt[i], len(x_plt)), "black", alpha=1, lw=2, ls="-.")
            # ax.set_xlim(0, len(samples))
            ax.set_title(labels[i])
            # ax.yaxis.set_label_coords(-0.1, 0.5)
            sns.despine(ax=ax, trim=False)
            ax.set_xlim(frompt, frompt + samples[frompt:].shape[0])

        axes[-1].set_xlabel("step number")

    plot_chain(0, axes[:, 0])
    plot_chain(3 * samples.shape[0] // 4, axes[:, 1])
    fig.tight_layout()
    if save_to is not None:
        plt.savefig(save_to)


def plot_fit(modinf, loss):
    subpop_names = modinf.subpop_struct.subpop_names
    fig, axes = plt.subplots(
        len(subpop_names), len(loss.statistics), figsize=(3 * len(loss.statistics), 3 * len(subpop_names)), sharex=True
    )
    for j, subpop in enumerate(modinf.subpop_struct.subpop_names):
        gt_s = loss.gt[loss.gt["subpop"] == subpop].sort_index()
        first_date = max(gt_s.index.min(), results[0].index.min())
        last_date = min(gt_s.index.max(), results[0].index.max())
        gt_s = gt_s.loc[first_date:last_date].drop(["subpop"], axis=1).resample("W-SAT").sum()

        for i, (stat_name, stat) in enumerate(loss.statistics.items()):
            ax = axes[j, i]

            ax.plot(gt_s[stat.data_var], color="k", marker=".", lw=1)
            for model_df in results:
                model_df_s = (
                    model_df[model_df["subpop"] == subpop]
                    .drop(["subpop"], axis=1)
                    .loc[first_date:last_date]
                    .resample("W-SAT")
                    .sum()
                )  # todo sub subpop here
                ax.plot(model_df_s[stat.sim_var], lw=0.9, alpha=0.5)
            # if True:
            #        init_df_s = outcomes_df_ref[model_df["subpop"]==subpop].drop(["subpop","time"],axis=1).loc[min(gt_s.index):max(gt_s.index)].resample("W-SAT").sum() # todo sub subpop here
            ax.set_title(f"{stat_name}, {subpop}")
    fig.tight_layout()
    plt.savefig(f"{run_id}_results.pdf")

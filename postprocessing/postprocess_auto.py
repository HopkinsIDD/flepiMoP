import gempyor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import glob, os, sys
from pathlib import Path

# import seaborn as sns
import matplotlib._color_data as mcd
import pyarrow.parquet as pq
import click

import dask.dataframe as dd
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from matplotlib.backends.backend_pdf import PdfPages

channelids = {"cspproduction": "C011YTUBJ7R", "debug": "C04MAQWLEAW"}


class RunInfo:
    def __init__(self, run_id, config_filepath=None, folder_path=None):
        self.run_id = run_id
        self.config_filepath = config_filepath
        self.folder_path = folder_path


def get_all_filenames(
    file_type, all_runs, finals_only=False, intermediates_only=False, ignore_chimeric=True
) -> dict:
    """
    return dictionanary for each run name
    """
    if file_type == "seed":
        ext = "csv"
    else:
        ext = "parquet"
    files = {}
    for run_name, run_info in all_runs.items():
        l = []
        for f in Path(str(run_info.folder_path)).rglob(f"*.{ext}"):
            f = str(f)
            if file_type in f:
                if (
                    (finals_only and "final" in f)
                    or (intermediates_only and "intermediate" in f)
                    or (not finals_only and not intermediates_only)
                ):
                    if not (ignore_chimeric and "chimeric" in f):
                        l.append(str(f))
        files[run_name] = l
    return files


def slack_multiple_files_deprecated(slack_token, message, file_list, channel):
    import logging
    from slack_sdk import WebClient

    client = WebClient(slack_token)
    logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG)
    for file in file_list:
        upload = client.files_upload(file=file, filename=file)
        message = message + "<" + upload["file"]["permalink"] + "| >"
    outP = client.chat_postMessage(channel=channel, text=message)


def slack_multiple_files_v2(slack_token, message, file_list, channel):
    # file_uploads=[
    #    {
    #        "file": "pplot_llik_FCH_R3_highVE_pesImm_2022_Jan22_USA-20230130T163847_inference_med.pdf",
    #        "title": "Log-likelihood plot",
    #    },
    #    {
    #        "file": "slurm-11598936_237.out",
    #        "title": "random log file",
    #    },
    # ],
    import logging
    from slack_sdk import WebClient

    client = WebClient(slack_token)
    logging.basicConfig(level=logging.DEBUG)
    file_uploads = [{"file": f, "title": f.split(".")[0]} for f in file_list]
    response = client.files_upload_v2(
        file_uploads=file_uploads,
        channel=channel,
        initial_comment=message,
    )
    return response


@click.command()
@click.option(
    "-c",
    "--config",
    "config_filepath",
    envvar="CONFIG_PATH",
    type=click.Path(exists=True),
    required=True,
    help="configuration file for this simulation",
)
@click.option(
    "-I",
    "--run-id",
    "run_id",
    envvar="FLEPI_RUN_INDEX",
    type=str,
    show_default=True,
    help="run index",
)
@click.option(
    "-j",
    "--job-name",
    "job_name",
    envvar="JOB_NAME",
    default="x",
    type=str,
    show_default=True,
    help="unique identifier for the run",
)
@click.option(  # slurm only option
    "-f",
    "--fs-results-path",
    "fs_results_path",
    envvar="FS_RESULTS_PATH",
    type=click.Path(exists=True),
    default=".",
    show_default=True,
    help="The file system folder to use load the simulations from",
)
@click.option(  # slurm only option
    "-s",
    "--slack-token",
    "slack_token",
    envvar="SLACK_TOKEN",
    type=str,
    help="Slack token",
)
@click.option(
    "-s",
    "--slack-channel",
    "slack_channel",
    envvar="SLACK_CHANNEL",
    type=str,
    help="Slack channel, either 'csp-production' or 'debug'",
)
@click.option(
    "-m",
    "--max_files",
    type=click.IntRange(min=1),
    default=90000,
    help="Maximum number of files to load for aggregate plot, e.g quantiles",
)
@click.option(
    "-M",
    "--max_files_deep",
    type=click.IntRange(min=1),
    default=30,
    help="Maximum number of files to load for in depth plot and individual sim plot",
)
def generate_pdf(
    config_filepath,
    run_id,
    job_name,
    fs_results_path,
    slack_token,
    slack_channel,
    max_files,
    max_files_deep,
):
    print("Generating plots")
    print(f">> config {config_filepath} for run_id {run_id}")
    print(f">> job name {job_name}, path {fs_results_path}")
    print(f">> max files (normal, deeep): {max_files}, {max_files_deep}")

    try:
        all_runs = {
            run_id: RunInfo(run_id, config_filepath),
        }

        # In[4]:

        for run_name, run_info in all_runs.items():
            run_id = run_info.run_id
            config_filepath = run_info.config_filepath
            run_info.gempyor_inference = gempyor.GempyorInference(
                config_filepath=config_filepath,
                run_id=run_id,
                # prefix=f"USA/inference/med/{run_id}/global/intermediate/000000001.",
                first_sim_index=1,
                path_prefix="./",  # prefix where to find the folder indicated in subpop_setup$
            )
            run_info.folder_path = f"{fs_results_path}/model_output"

        node_names = run_info.gempyor_inference.modinf.subpop_struct.subpop_names

        # In[5]:

        # gempyor.config.set_file(run_info.config_filepath)
        # gt = pd.read_csv(gempyor.config["inference"]["gt_data_path"].get())
        # gt
        # statistics = {}
        # Ingoring agreegation and all, assuming by week
        # for stat in gempyor.config["inference"]["statistics"]:
        #    statistics[gempyor.config["inference"]["statistics"][stat]["sim_var"].get()] = gempyor.config["inference"][
        #        "statistics"
        #    ][stat]["data_var"].get()
        # statistics

        # ## Analyze llik files

        # In[6]:

        llik_filenames = get_all_filenames("llik", all_runs, intermediates_only=True)

        # In[7]:

        resultST = {}

        for run_name, run_info in all_runs.items():
            resultST[run_name] = []
            file_list = llik_filenames[run_name][:max_files]
            for filename in file_list:
                slot = int(filename.split("/")[-1].split(".")[0])
                block = int(filename.split("/")[-1].split(".")[1])
                sim_str = filename.split("/")[-1].split(".")[
                    2
                ]  # not necessarily a sim number now
                if sim_str.isdigit():
                    sim = int(sim_str)
                    if block == 1 and (sim == 1 or sim % 5 == 0):  ## first block, only one
                        df_raw = pq.read_table(filename).to_pandas()
                        df_raw["slot"] = slot
                        df_raw["sim"] = sim
                        df_raw["ID"] = run_name
                        df_raw = df_raw.drop("filename", axis=1)
                        # df_csv = df_csv.groupby(['slot','sim', 'ID', 'subpop']).sum().reset_index()
                        # df_csv = df_csv[['ll','sim', 'slot', 'ID','subpop']]
                        resultST[run_name].append(df_raw)
        full_df = pd.concat(resultST[run_name])
        full_df

        # In[22]:

        full_df.groupby(["sim", "slot"]).sum()

        # In[23]:

        fig, axes = plt.subplots(
            len(node_names) + 1, 4, figsize=(4 * 4, len(node_names) * 3), sharex=True
        )

        colors = ["b", "r", "y", "c"]
        icl = 0

        idp = 0
        all_nn = (
            full_df.groupby(["sim", "slot"])
            .sum()
            .reset_index()[["sim", "slot", "ll", "accept", "accept_avg", "accept_prob"]]
        )
        for ift, feature in enumerate(["ll", "accept", "accept_avg", "accept_prob"]):
            lls = all_nn.pivot(index="sim", columns="slot", values=feature)
            if feature == "accept":
                lls = lls.cumsum()
                feature = "accepts, cumulative"
            axes[idp, ift].fill_between(
                lls.index,
                lls.quantile(0.025, axis=1),
                lls.quantile(0.975, axis=1),
                alpha=0.1,
                color=colors[icl],
            )
            axes[idp, ift].fill_between(
                lls.index,
                lls.quantile(0.25, axis=1),
                lls.quantile(0.75, axis=1),
                alpha=0.1,
                color=colors[icl],
            )
            axes[idp, ift].plot(
                lls.index, lls.median(axis=1), marker="o", label=run_id, color=colors[icl]
            )
            axes[idp, ift].plot(lls.index, lls.iloc[:, 0:max_files_deep], color="k", lw=0.3)
            axes[idp, ift].set_title(f"National, {feature}")
            axes[idp, ift].grid()

        for idp, nn in enumerate(node_names):
            idp = idp + 1
            all_nn = full_df[full_df["subpop"] == nn][
                ["sim", "slot", "ll", "accept", "accept_avg", "accept_prob"]
            ]
            for ift, feature in enumerate(["ll", "accept", "accept_avg", "accept_prob"]):
                lls = all_nn.pivot(index="sim", columns="slot", values=feature)
                if feature == "accept":
                    lls = lls.cumsum()
                    feature = "accepts, cumulative"
                axes[idp, ift].fill_between(
                    lls.index,
                    lls.quantile(0.025, axis=1),
                    lls.quantile(0.975, axis=1),
                    alpha=0.1,
                    color=colors[icl],
                )
                axes[idp, ift].fill_between(
                    lls.index,
                    lls.quantile(0.25, axis=1),
                    lls.quantile(0.75, axis=1),
                    alpha=0.1,
                    color=colors[icl],
                )
                axes[idp, ift].plot(
                    lls.index,
                    lls.median(axis=1),
                    marker="o",
                    label=run_id,
                    color=colors[icl],
                )
                axes[idp, ift].plot(
                    lls.index, lls.iloc[:, 0:max_files_deep], color="k", lw=0.3
                )
                axes[idp, ift].set_title(f"{nn}, {feature}")
                axes[idp, ift].grid()
                if idp == len(node_names) - 1:
                    axes[idp, ift].set_xlabel("sims")
                # ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
        fig.tight_layout()
        plt.savefig(f"pplot/llik_{run_id}_{job_name}.pdf")
    except:
        pass

    import gempyor.utils

    llik_filenames = gempyor.utils.list_filenames(
        folder="model_output/", filters=["final", "llik", ".parquet"]
    )
    # get_all_filenames("llik", fs_results_path, finals_only=True, intermediates_only=False)
    # In[7]:
    resultST = []
    for filename in llik_filenames:
        slot = int(filename.split("/")[-1].split(".")[0])
        df_raw = pq.read_table(filename).to_pandas()
        df_raw["slot"] = slot
        df_raw["filename"] = filename  # so it contains the /final/ filename
        resultST.append(df_raw)

    full_df = pd.concat(resultST).set_index(["slot"])
    sorted_llik = full_df.groupby(["slot"]).sum().sort_values("ll", ascending=False)
    fig, axes = plt.subplots(1, 1, figsize=(5, 10))
    # ax = axes.flat[0]
    ax = axes
    ax.plot(sorted_llik["ll"].reset_index(drop=True), marker=".")
    ax.set_xlabel("slot (sorted by llik)")
    ax.set_ylabel("llik")
    ax.set_title("llik by slot")
    # vertical line at cutoff
    # log scale in axes two:
    # ax = axes.flat[1]
    # ax.plot(sorted_llik["ll"].reset_index(drop=True))
    # ax.set_xlabel("slot")
    # ax.set_ylabel("llik")
    # ax.set_title("llik by slot (log scale)")
    # ax.set_yscale("log")
    ## vertical line at cutoff
    # ax.axvline(x=best_n, color="red", linestyle="--")
    ax.grid()
    plt.show()
    plt.savefig("pplot/llik_by_slot.png")

    file_list = []
    # f or f in Path(str(".")).rglob(f"./pplot/*"): # this took all the files also very deep into subdirectories
    for f in glob.glob(f"pplot/*"):
        file_list.append(str(f))

    print(f"list of files to be sent over slack: {file_list}")

    if "production" in slack_channel.lower():
        channel = channelids["cspproduction"]
    elif "debug" in slack_channel.lower():
        channel = channelids["debug"]
    else:
        print("no channel specified, not sending anything to slack")
        channel = None

    # slack_multiple_files(
    #    slack_token=slack_token,
    #    message=f"FlepiMoP run `{run_id}` (job `{job_name}`) has successfully completed 🎉🤖. \n \nPlease find below a little analysis of the llik files, and I'll try to be more helpful in the future.",
    #    fileList=flist,
    #    channel=channelid_chadi,
    # )
    if channel is not None:
        r = slack_multiple_files_v2(
            slack_token=slack_token,
            message=f"""FlepiMoP run `{run_id}` (job `{job_name}`) has successfully completed 🎉🤖. \n \nPlease find below a little analysis of the llik files, and I'll try to be more helpful in the future.""",
            file_list=file_list,
            channel=channel,
        )
        print(f"api response: {r}")


if __name__ == "__main__":
    generate_pdf()

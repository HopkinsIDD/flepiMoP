import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import glob, os, sys
from pathlib import Path

import pyarrow.parquet as pq
import click



def get_all_filenames(file_type, fs_results_path="to_prune/",  finals_only=False, intermediates_only=True, ignore_chimeric=True) -> dict:
    """
      return dictionanary for each run name
    """
    if file_type=="seed":
        ext="csv"
    else:
        
        ext="parquet"
    l = []
    for f in Path(str(fs_results_path + "model_output")).rglob(f'*.{ext}'):
        f = str(f)
        if file_type in f:
            if (finals_only and "final" in f) or (intermediates_only and "intermediate" in f) or (not finals_only and not intermediates_only):
                if not (ignore_chimeric and "chimeric" in f):
                    l.append(str(f))
    return l


@click.command()
@click.option(
    "-f",
    "--fs-results-path",
    "fs_results_path",
    envvar="FS_RESULTS_PATH",
    type=click.Path(exists=True),
    default="to_prune/",
    show_default=True,
    help="The file system folder to use load the simulations from",
)
@click.option(
    "-n",
    "--best-n",
    "best_n"
    type=click.IntRange(min=1),
    default=10,
    help="Duplicate the best n files (default 10)",
)

def generate_pdf(fs_results_path, best_n):
    print("pruning by llik")
    fs_results_path = "to_prune/"
    best_n = 265

    llik_filenames = get_all_filenames("llik", fs_results_path ,finals_only=True)

    # In[7]:

    resultST = []

    for filename in llik_filenames:
        slot = int(filename.split("/")[-1].split(".")[0])
        df_raw = pq.read_table(filename).to_pandas()
        df_raw["slot"] = slot
        df_raw["filename"] = filename # so it contains the /final/ filename
        resultST.append(df_raw)
    full_df = pd.concat(resultST).set_index(["slot"])

    sorted_llik = full_df.groupby(["slot"]).sum().sort_values("ll", ascending=False)
    best_slots = sorted_llik.head(best_n).index.values


    fig, axes = plt.subplots(1, 1, figsize=(5, 10))
    #ax = axes.flat[0]
    ax = axes
    ax.plot(sorted_llik["ll"].reset_index(drop=True), marker = ".")
    ax.set_xlabel("slot (sorted by llik)")
    ax.set_ylabel("llik")
    ax.set_title("llik by slot")
    # vertical line at cutoff
    ax.axvline(x=best_n, color="red", linestyle="--")
    # log scale in axes two:
    #ax = axes.flat[1]
    #ax.plot(sorted_llik["ll"].reset_index(drop=True))
    #ax.set_xlabel("slot")
    #ax.set_ylabel("llik")
    #ax.set_title("llik by slot (log scale)")
    #ax.set_yscale("log")
    ## vertical line at cutoff
    #ax.axvline(x=best_n, color="red", linestyle="--")
    ax.grid()
    plt.show()
    plt.savefig("llik_by_slot.pdf")

    print(f"Top {best_n} slots by llik are:")
    for slot in best_slots:
        print(f" - {slot:4}, llik: {sorted_llik.loc[slot]['ll']:0.3f}")


    files_to_keep = list(full_df.loc[best_slots]["filename"].unique())

    all_files = list(full_df["filename"].unique())
    
    output_folder = "pruned/"


    def copy_path(src, dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        import shutil
        print(f"copying {src} to {dst}")
        shutil.copy(src, dst)

    file_types= ["llik", "seed", "snpi", "hnpi", "spar", "hpar", "init"] # TODO: init here but don't fail if not found

    for fn in all_files:
        print(f"processing {fn}")
        if fn in files_to_keep:
            for file_type in file_types:
                src = fn.replace("llik", file_type)
                dst = fn.replace(fs_results_path, output_folder).replace("llik", file_type)
                if file_type == "seed":
                    src = src.replace(".parquet", ".csv")
                    dst = dst.replace(".parquet", ".csv")
                copy_path(src=src, dst=dst)
        else:
            file_to_keep = np.random.choice(files_to_keep)
            for file_type in file_types:
                src = file_to_keep.replace("llik", file_type)
                dst = fn.replace(fs_results_path, output_folder).replace("llik", file_type)
                if file_type == "seed":
                    src = src.replace(".parquet", ".csv")
                    dst = dst.replace(".parquet", ".csv")
                copy_path(src=src, dst=dst) 



if __name__ == "__main__":
    generate_pdf()
import abc
import pyarrow as pa
import click


class NPIBase(abc.ABC):
    __plugins__ = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        NPIBase.__plugins__[cls.__name__] = cls

    def __init__(self, *, name):
        self.name = name

    @abc.abstractmethod
    def getReduction(self, param, default=None):
        pass

    # Returns dataframe with columns: <subpops>, time, parameter, name. Index is sequential.
    @abc.abstractmethod
    def getReductionToWrite(self):
        pass

    def getReductionDF(self):
        return self.getReductionToWrite()

    def execute(
        *,
        npi_config,
        modinf,
        modifiers_library,
        subpops,
        loaded_df=None,
        pnames_overlap_operation_sum=[],
        pnames_overlap_operation_reductionprod=[],
    ):
        """
        npi_config: config of the Modifier we are building, usually a StackedModifiers that will call other NPI
        modinf: the ModelInfor class, to inform ti and tf
        modifiers_library: a config bit that contains the other modifiers that could be called by this Modifier. Note
            that the confuse library's config resolution mechanism makes slicing the configuration object expensive;
            instead give the preloaded settings from .get()
        """
        method = npi_config["method"].as_str()
        npi_class = NPIBase.__plugins__[method]
        return npi_class(
            npi_config=npi_config,
            modinf=modinf,
            modifiers_library=modifiers_library,
            subpops=subpops,
            loaded_df=loaded_df,
            pnames_overlap_operation_sum=pnames_overlap_operation_sum,
            pnames_overlap_operation_reductionprod=pnames_overlap_operation_reductionprod,
        )
    

@click.group()
def modifiers():
    pass


# TODO: CLI arguments
@modifiers.command()
@click.option(
    "-c",
    "--config",
    "config_filepath",
    envvar=["CONFIG_PATH"],
    type=click.Path(exists=True),
    help="configuration file for this simulation",
) # This is very bad because already set in cli.py, but ok....
@click.option(
    "-p",
    "--project_path",
    "project_path",
    envvar="PROJECT_PATH",
    type=click.Path(exists=True),
    default=".",
    required=True,
    help="path to the flepiMoP directory",
)
@click.option(
    "--id",
    "--id",
    "run_id",
    envvar="FLEPI_RUN_INDEX",
    type=str,
    default=None,
    help="Unique identifier for this run",
)
@click.option('--nsamples', default=30, help='Number of samples to draw')
@click.option('--subpop', default=None, help='Subpopulation to plot, if not set then all are plotted')
def config_plot(config_filepath, project_path, run_id, nsamples, subpop):
    from gempyor.inference import GempyorInference
    import gempyor
    import os
    from gempyor.utils import config
    if run_id is None:
        import pathlib
        base_run_id = pathlib.Path(config_filepath).stem.replace("config_", "")
        run_id = f"{base_run_id}-{gempyor.file_paths.run_id()}"

    # Config prep
    config.clear()
    config.read(user=False)

    config.set_file(os.path.join(project_path, config_filepath))
    gempyor_inference = GempyorInference(
        config_filepath=config_filepath,
        run_id=run_id,
        prefix=None,
        first_sim_index=1,
        stoch_traj_flag=False,
        rng_seed=None,
        nslots=1,
        inference_filename_prefix="global/final/",  # usually for {global or chimeric}/{intermediate or final}
        inference_filepath_suffix="",  # usually for the slot_id
        out_run_id=None,  # if out_run_id is different from in_run_id, fill this
        out_prefix=None,  # if out_prefix is different from in_prefix, fill this
        path_prefix=project_path,  # in case the data folder is on another directory
        autowrite_seir=False,
)
    if gempyor_inference.modinf.seir_config is not None and gempyor_inference.modinf.npi_config_seir is not None:
        npi_seir = gempyor.seir.build_npi_SEIR(
            modinf=gempyor_inference.modinf, 
            load_ID=False,
            sim_id2load=None,
            config=config
            )
    if gempyor_inference.modinf.outcomes_config is not None and gempyor_inference.modinf.npi_config_outcomes:
        npi_outcomes = gempyor.outcomes.build_outcome_modifiers(
            modinf=gempyor_inference.modinf,
            load_ID=False,
            sim_id2load=None,
            config=config,
        )


    print("Plotting modifiers activation")
    if subpop is None:
        subpop =  gempyor_inference.modinf.subpop_struct.subpop_names
    elif isinstance(subpop, str):
        subpop = [subpop]
    for sp in subpop:
        plot_modifers_activation(npi_seir=npi_seir,
                            filename=f"seir_modifiers_activation_{sp}.pdf",
                                subpop=sp)
        plot_modifers_activation(npi_seir=npi_outcomes,
                    filename=f"outcomes_modifiers_activation_{sp}.pdf",
                    subpop=sp)
    
    all_parsed_params_seir = []
    for sample in range(nsamples):
        if True:
            p_draw = gempyor_inference.modinf.parameters.parameters_quick_draw(
                n_days=gempyor_inference.modinf.n_days, 
                nsubpops=gempyor_inference.modinf.nsubpops
            )
            npi_seir = gempyor.seir.build_npi_SEIR(
                modinf=gempyor_inference.modinf, 
                load_ID=False,
                sim_id2load=None,
                config=config
            )
            npi_outcomes = gempyor.outcomes.build_outcome_modifiers(
            modinf=gempyor_inference.modinf,
            load_ID=False,
            sim_id2load=None,
            config=config,
        )

        else:
            p_draw = gempyor_inference.get_seir_parameters(bypass_FN=fn.replace("snpi", "spar"), load_ID=True)
            npi_seir = gempyor.seir.build_npi_SEIR(modinf=gempyor_inference.modinf, load_ID=True, bypass_FN=fn, sim_id2load=None, config=None)
        parameters = gempyor_inference.modinf.parameters.parameters_reduce(p_draw, npi_seir)
        parsed_parameters = gempyor_inference.modinf.compartments.parse_parameters(
            parameters, 
            gempyor_inference.modinf.parameters.pnames, 
            gempyor_inference.static_sim_arguments["unique_strings"]
        )
        all_parsed_params_seir.append(parsed_parameters)
            

        

    print("Plotting parameter timeseries from config")
    plot_parameter_timeseries(all_parsed_params=all_parsed_params_seir, 
                            gempyor_inference=gempyor_inference, 
                            filename=f"unique_parsed_parameters_{run_id}.pdf")
    
    npi_outcomes = gempyor.outcomes.build_outcome_modifiers(
            modinf=gempyor_inference.modinf,
            load_ID=False,
            sim_id2load=None,
            config=config,
        )
    plot_npi_timeseries(npi_outcomes, gempyor_inference, filename=f"outcomesNPIcaveat.pdf", nsamples=nsamples)



        

def plot_modifers_activation(npi_seir, filename: str, subpop: str):
    # TODO should return axes and all
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime
    from matplotlib.backends.backend_pdf import PdfPages

# Helper function to convert date strings to datetime objects
    def date_to_datetime(date_str):
        date_format = "%Y-%m-%d"
        return datetime.strptime(date_str, date_format)

    with PdfPages(filename) as pdf:
        d = pdf.infodict()
        d["Title"] = "FlepiMoP Inference Fit"
        d["Author"] = "FlepiMoP Inference"

        for param in npi_seir.getReductionDF()["parameter"].unique():
            data = npi_seir.getReductionDF()

            data = data[(data["parameter"]==param)]
            data = data[data.apply(lambda row: row.astype(str).str.contains(subpop).any(), axis=1)]


            # Parse the data into lists of datetime objects
            parsed_data = []
            for i, entry in data.iterrows():
                starts = entry["start_date"].split(',')
                ends = entry["end_date"].split(',')
                parsed_data.append({
                    'modifier_name': entry['modifier_name'],
                    'start_times': [date_to_datetime(date) for date in starts],
                    'end_times': [date_to_datetime(date) for date in ends]
                })

            # Create the figure and axis
            fig, ax = plt.subplots(figsize=(10, 3))  # Adjust figure size as needed

            # Define the y-axis positions for each element (equal spacing for each element)
            elements = [entry['modifier_name'] for entry in parsed_data]
            y_positions = range(len(elements))  # Y positions for each element

            # Plot horizontal lines for each element's active periods
            for i, element_data in enumerate(parsed_data):
                starts = element_data['start_times']
                ends = element_data['end_times']
                
                # Plot each active period as a horizontal line
                for start, end in zip(starts, ends):
                    ax.hlines(y=i, xmin=start, xmax=end, color=f'C{i}', linewidth=5)

            # Set the y-ticks to show element names (modifiers)
            ax.set_yticks(y_positions)
            ax.set_yticklabels(elements)

            # Format the x-axis to show dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Date format for x-axis
            ax.xaxis.set_major_locator(mdates.MonthLocator())  # Show one tick per month
            fig.autofmt_xdate()  # Rotate date labels for better visibility

            # Add labels and grid
            ax.set_xlabel('Date')
            ax.set_ylabel('Modifier')
            ax.set_title(f'Activation Time Periods of Modifiers for {param} in {subpop}')

            # Add grid for clarity
            ax.grid(True, which='both', axis='x', linestyle='--', alpha=0.7)
            pdf.savefig(fig)

def plot_parameter_timeseries(all_parsed_params, gempyor_inference, filename: str):
    # TODO should return axes and all
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime
    from matplotlib.backends.backend_pdf import PdfPages
    import pandas as pd

    pdf = PdfPages(filename)
    d = pdf.infodict()
    d["Title"] = "parsed parameters"
    d["Author"] = "jlemaitre"
    import tqdm

    for k,uniq_pname in tqdm.tqdm(enumerate(gempyor_inference.static_sim_arguments["unique_strings"])):
        #if 'r0*gamma*phi_latino*theta1_IOTA*chi_IOTA*1*phi_white' in uniq_pname:
        if True: # we need to add filtering and prine here.
            fig, axes = plt.subplots(len(gempyor_inference.modinf.subpop_struct.subpop_names), 1, 
                                    figsize=(10, len(gempyor_inference.modinf.subpop_struct.subpop_names)*3), 
                                    sharex=True, sharey=True)
            fig.suptitle(uniq_pname, fontsize=22)
            #print(uniq_pname)
            for i, geoid in enumerate(gempyor_inference.modinf.subpop_struct.subpop_names):
                if len(gempyor_inference.modinf.subpop_struct.subpop_names) == 1:
                    ax = axes
                else:
                    ax = axes.flat[i]
                ax.set_title(geoid)
                ax.grid()
                for l in range(len(all_parsed_params)):
                    df = pd.DataFrame(all_parsed_params[l][k,:,i], index=pd.date_range(gempyor_inference.modinf.ti, 
                                                                                    gempyor_inference.modinf.tf, 
                                                                                    freq="D"))
                    ax.plot(df, lw=.5) #df[:'2021-12-31']
                fig.autofmt_xdate()
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    pdf.close()

def plot_npi_timeseries(npi, gempyor_inference, filename: str, nsamples:int):
    # TODO should return axes and all
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime
    from matplotlib.backends.backend_pdf import PdfPages
    import pandas as pd
    from gempyor.NPI.helpers import reduce_parameter

    pdf = PdfPages(filename)
    d = pdf.infodict()
    d["Title"] = "parsed parameters"
    d["Author"] = "jlemaitre"
    import tqdm

    for k,uniq_pname in tqdm.tqdm(enumerate(npi.getReductionDF()["parameter"].unique())):
        fig, axes = plt.subplots(len(gempyor_inference.modinf.subpop_struct.subpop_names), 1, 
                                figsize=(10, len(gempyor_inference.modinf.subpop_struct.subpop_names)*3), 
                                sharex=True, sharey=True)
        fig.suptitle(uniq_pname, fontsize=22)
        #print(uniq_pname)
        for i, geoid in enumerate(gempyor_inference.modinf.subpop_struct.subpop_names):
            if len(gempyor_inference.modinf.subpop_struct.subpop_names) == 1:
                ax = axes
            else:
                ax = axes.flat[i]
            ax.set_title(geoid)
            ax.grid()
            for sample in range(nsamples):
                # todo nothing is correct here
                rd = reduce_parameter(1,
                    modification=npi.getReduction(uniq_pname),
                )
                df = pd.DataFrame(rd[:,i], index=pd.date_range(gempyor_inference.modinf.ti, 
                                                                                gempyor_inference.modinf.tf, 
                                                                                freq="D"))
                ax.plot(df, lw=.5) #df[:'2021-12-31']
            fig.autofmt_xdate()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    pdf.close()
import abc
import pyarrow as pa


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

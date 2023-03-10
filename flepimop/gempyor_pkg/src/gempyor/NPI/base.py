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

    # Returns dataframe with columns: <geoids>, time, parameter, name. Index is sequential.
    @abc.abstractmethod
    def getReductionToWrite(self):
        pass

    def getReductionDF(self):
        return self.getReductionToWrite()

    def execute(
        *,
        npi_config,
        global_config,
        geoids,
        loaded_df=None,
        pnames_overlap_operation_sum=[],
    ):
        template = npi_config["template"].as_str()
        npi_class = NPIBase.__plugins__[template]
        return npi_class(
            npi_config=npi_config,
            global_config=global_config,
            geoids=geoids,
            loaded_df=loaded_df,
            pnames_overlap_operation_sum=pnames_overlap_operation_sum,
        )

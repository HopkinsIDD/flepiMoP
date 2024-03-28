class SimulationComponent:
    def __init__(self, config: confuse.ConfigView):
        raise NotImplementedError("This method should be overridden in subclasses.")

    def get_from_file(self, sim_id: int, setup) -> np.ndarray:
        raise NotImplementedError("This method should be overridden in subclasses.")

    def get_from_config(self, sim_id: int, setup) -> np.ndarray:
        raise NotImplementedError("This method should be overridden in subclasses.")

    def write_to_file(self, sim_id: int, setup):
        raise NotImplementedError("This method should be overridden in subclasses.")
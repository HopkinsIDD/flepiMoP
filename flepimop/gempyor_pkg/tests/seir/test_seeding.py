import os
from gempyor import seeding_ic, model_info
from gempyor.utils import config

DATA_DIR = os.path.dirname(__file__) + "/data"
os.chdir(os.path.dirname(__file__))


class TestSeeding:
    def test_Seeding_success(self):
        config.clear()
        config.read(user=False)
        config.set_file(f"{DATA_DIR}/config.yml")

        s = model_info.ModelInfo(
            config=config,
            setup_name="test_seeding",
            nslots=1,
            seir_modifiers_scenario=None,
            outcome_modifiers_scenario=None,
            write_csv=False,
        )
        sic = seeding_ic.Seeding(config=s.seeding_config)
        assert sic.seeding_config == s.seeding_config

    def test_Seeding_draw_success(self):
        config.clear()
        config.read(user=False)
        config.set_file(f"{DATA_DIR}/config.yml")

        s = model_info.ModelInfo(
            config=config,
            setup_name="test_seeding",
            nslots=1,
            seir_modifiers_scenario=None,
            outcome_modifiers_scenario=None,
            write_csv=False,
        )
        sic = seeding_ic.Seeding(config=s.seeding_config)
        s.seeding_config["method"] = "NoSeeding"

        seeding = sic.draw(sim_id=100, setup=s)
        print(seeding)

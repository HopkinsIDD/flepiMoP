# Cross platform customizations
ifeq ($(OS),Windows_NT)
    RM_CMD = del /Q
    RM_DIR = rmdir /S /Q
else
    RM_CMD = rm -f
    RM_DIR = rm -rf
endif


# Default target to build all outputs
.PHONY: all
all: \
	model_output/three_state_seir_stacked_hosp_stacked/emcee_inference \
	model_output/three_state_seir_stacked_hosp_stacked/r_inference


# Clean all generated files & targets
.PHONY: clean
clean:
	$(RM_DIR) model_output
	$(RM_CMD) model_input/ground_truth_hospitalizations.csv
	$(RM_CMD) three_state_simulate.yml
	$(RM_CMD) three_state_emcee_inference.yml
	$(RM_CMD) three_state_r_inference.yml
	$(RM_CMD) log*.txt
	$(RM_CMD) *.h5
	$(RM_CMD) *.pdf


# Simulate ground truth data
three_state_simulate.yml:
	flepimop patch \
		sim_base.yml \
		--method stochastic \
		--nslots 1 \
		--in-id sim > three_state_simulate.yml

model_output/three_state_state_varied_Ro_state_varied_incidH/sim: three_state_simulate.yml
	flepimop simulate \
		--seir_modifiers_scenarios state_varied_Ro \
		--outcome_modifiers_scenarios state_varied_incidH \
		three_state_simulate.yml

model_input/ground_truth_hospitalizations.csv: model_output/three_state_state_varied_Ro_state_varied_incidH/sim
	python ground_truth_from_simulation.py


# Inference with EMCEE
three_state_emcee_inference.yml:
	flepimop patch \
		inference_base.yml \
		inference_outcome_modifiers.yml \
		inference_seir_modifiers.yml \
		emcee_inference.yml \
		> three_state_emcee_inference.yml

model_output/three_state_seir_stacked_hosp_stacked/emcee_inference: \
	three_state_emcee_inference.yml model_input/ground_truth_hospitalizations.csv
	flepimop-calibrate \
		--config three_state_emcee_inference.yml \
		--jobs 12 \
		--nslots 12 \
		--niterations 500 \
		--nsamples 100 \
		--id emcee_inference


# Inference with R
three_state_r_inference.yml:
	flepimop patch \
		inference_base.yml \
		inference_outcome_modifiers.yml \
		inference_seir_modifiers.yml \
		r_inference.yml \
		> three_state_r_inference.yml

model_output/three_state_seir_stacked_hosp_stacked/r_inference: \
	three_state_r_inference.yml model_input/ground_truth_hospitalizations.csv
	flepimop-inference-main \
		--config three_state_r_inference.yml \
		--run_id r_inference \
		--seir_modifiers_scenarios seir_stacked \
		--outcome_modifiers_scenarios hosp_stacked \
		--jobs 12 \
		--iterations_per_slot 600 \
		--slots 12 \
		--python "$(CONDA_PREFIX)/bin/python" \
		--rpath "$(CONDA_PREFIX)/bin/R" \
		--save_hosp TRUE \
		--save_seir TRUE

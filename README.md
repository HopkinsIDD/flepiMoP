
# FlepiMoP

The *Fle*xible *Epi*demic *Mo*deling *P*ipeline, `FlepiMoP`, makes it easy to build an infectious disease model, infer that model's parameters, and project scenario outcomes.

# Quickstart

```bash
mkdir myflepimopworkspace && cd $_
git clone git@github.com:HopkinsIDD/flepiMoP.git --depth 1
./flepiMop/install_ubuntu.sh
cp -r ./flepiMoP/examples/tutorial_two_subpops test_model && cd $_
gempyor-simulate -c config_sample_2pop.yml
flepimop-inference-main -c config_sample_2pop_inference.yml
```

# flepiMoP

Welcome to the Johns Hopkins University Infectious Disease Dynamics's `Flexible Epidemic Modeling Pipeline`. “FlepiMoP” provides a framework for quickly coding and simulating compartmental infectious disease models to project epidemic trajectories and their healthcare impacts, and to evaluate the impact of potential interventions. The package is a work-in-progress but is extensively documented https://iddynamics.gitbook.io/flepimop/, with instructions describing how to install the package, code up your model, run forward simulations, and infer model parameters from timeseries data. More details of the project are available on our dedicated website https://www.flepimop.org/. 

We recommend that most new users use the code from the stable `main` branch. Please post questions to GitHub issues with the `question` tag. We are prioritizing direct support for individuals engaged in public health planning and emergency response.

This open-source project is licensed under GPL v3.0.

Details on the methods and features of our model as of December 2023 were published in [Lemaitre JC, et al. flepiMoP: The evolution of a flexible infectious disease modeling pipeline during the COVID-19 pandemic. Epidemics. 2024;47:100753](https://www.sciencedirect.com/science/article/pii/S1755436524000148). Please cite this paper if you use flepiMoP in any of your own publications. 



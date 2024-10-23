# Table of contents

* [Home](README.md)

## 🦠 gempyor: modeling infectious disease dynamics <a href="#gempyor" id="gempyor"></a>

* [Modeling infectious disease dynamics](gempyor/model-description.md)
* [Model Implementation](gempyor/model-implementation/README.md)
  * [flepiMoP's configuration file](gempyor/model-implementation/introduction-to-configuration-files.md)
  * [Specifying population structure](gempyor/model-implementation/specifying-population-structure.md)
  * [Specifying compartmental model](gempyor/model-implementation/compartmental-model-structure.md)
  * [Specifying initial conditions](gempyor/model-implementation/specifying-initial-conditions.md)
  * [Specifying seeding](gempyor/model-implementation/specifying-seeding.md)
  * [Specifying observational model](gempyor/model-implementation/outcomes-for-compartments.md)
  * [Distributions](gempyor/model-implementation/distributions.md)
  * [Specifying time-varying parameter modifications](gempyor/model-implementation/intervention-templates.md)
  * [Other configuration options](gempyor/model-implementation/other-configuration-options.md)
  * [Code structure](gempyor/model-implementation/code-structure.md)
* [Model Output](gempyor/output-files.md)

## 📈 Model Inference

* [Inference Description](model-inference/inference-description.md)
* [Inference Implementation](model-inference/inference-implementation/README.md)
  * [Specifying data source and fitted variables](model-inference/inference-implementation/specifying-data-source-and-fitted-variables.md)
  * [(OLD) Configuration options](model-inference/inference-implementation/configuration-options.md)
  * [(OLD) Configuration setup](model-inference/inference-implementation/old-configuration-setup.md)
  * [Code structure](model-inference/inference-implementation/code-structure.md)
* [Inference Model Output](model-inference/inference-model-output.md)

## 🖥️ More

* [Setting up the model and post-processing](more/setting-up-the-model-and-post-processing/README.md)
  * [Config writer](more/setting-up-the-model-and-post-processing/config-writer.md)
  * [Diagnostic plotting scripts](more/setting-up-the-model-and-post-processing/plotting-scripts.md)
  * [Create a post-processing script](more/setting-up-the-model-and-post-processing/create-a-post-processing-script.md)
  * [Reporting](more/setting-up-the-model-and-post-processing/reporting.md)
* [Advanced](more/advanced/README.md)
  * [File descriptions](more/advanced/file-descriptions.md)
  * [Numerical methods](more/advanced/numerical-methods.md)
  * [Additional parameter options](more/advanced/additional-parameter-options.md)
  * [Swapping model modules](more/advanced/swapping-model-modules.md)
  * [Resuming inference runs](more/advanced/resuming-inference-runs.md)
  * [Using plug-ins 🧩\[experimental\]](more/advanced/using-plug-ins-experimental.md)

## 🛠️ How To Run

* [Before any run](how-to-run/before-any-run.md)
* [Quick Start Guide](how-to-run/quick-start-guide.md)
* [Advanced run guides](how-to-run/advanced-run-guides/README.md)
  * [Running with Docker locally 🛳](how-to-run/advanced-run-guides/running-with-docker-locally.md)
  * [Running locally in a conda environment 🐍](how-to-run/advanced-run-guides/quick-start-guide-conda.md)
  * [Running on AWS 🌳](how-to-run/advanced-run-guides/running-on-aws.md)
  * [Running On A HPC With Slurm](how-to-run/advanced-run-guides/running-on-a-hpc-with-slurm.md)
* [Common errors](how-to-run/common-errors.md)
* [Useful commands](how-to-run/useful-commands.md)
* [Tips, tricks, FAQ](how-to-run/tips-tricks-faq.md)

## 🗜️ Development

* [Guidelines for contributors](development/python-guidelines-for-developers.md)

## Deprecated pages

* [Module specification](deprecated-pages/module-specification.md)

## JHU Internal

* [US specific How to Run](jhu-internal/us-specific-how-to-run/README.md)
  * [Running with Docker locally (outdated/US specific) 🛳](jhu-internal/us-specific-how-to-run/running-with-docker-locally.md)
  * [Running on Rockfish/MARCC - JHU 🪨🐠](jhu-internal/us-specific-how-to-run/slurm-submission-on-marcc.md)
  * [Running with docker on AWS - OLD probably outdated](jhu-internal/us-specific-how-to-run/running-with-docker-on-aws/README.md)
    * [Provisioning AWS EC2 instance](jhu-internal/us-specific-how-to-run/running-with-docker-on-aws/provisioning-aws-ec2-instance.md)
    * [AWS Submission Instructions: Influenza](jhu-internal/us-specific-how-to-run/running-with-docker-on-aws/aws-submission-instructions-influenza.md)
    * [AWS Submission Instructions: COVID-19](jhu-internal/us-specific-how-to-run/running-with-docker-on-aws/aws-submission-instructions-covid-19.md)
  * [Running with RStudio Server on AWS EC2](jhu-internal/us-specific-how-to-run/running-with-rstudio-server-on-aws-ec2.md)
* [Inference scratch](jhu-internal/inference-scratch.md)

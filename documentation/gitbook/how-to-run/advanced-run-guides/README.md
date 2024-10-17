# Advanced run guides

For running the model locally, especially for testing, non-inference runs, and short chains, we provide a guide for [setting up and running in a conda environment](quick-start-guide-conda.md), and [provide a Docker container for use](running-with-docker-locally.md). A Docker container is an environment which is isolated from the rest of the operating system i.e. you can create files, programs, delete and everything but that will not affect your OS. It is a local virtual OS within your OS. We recommend Docker for users who are not familiar with setting up environments and seek a containerized environment to quickly launch jobs ;

For longer inference runs across multiple slots, we provide instructions and scripts for two methods to launch on SLURM HPC and on AWS using Docker. These methods are best for launching large jobs (long inference chains, multi-core and computationally expensive model runs), but not the best methods for debugging model setups.

## Running locally

{% content-ref url="running-with-docker-locally.md" %}
[running-with-docker-locally.md](running-with-docker-locally.md)
{% endcontent-ref %}

{% content-ref url="quick-start-guide-conda.md" %}
[quick-start-guide-conda.md](quick-start-guide-conda.md)
{% endcontent-ref %}

## Running longer inference runs across multiple slots

{% content-ref url="slurm-submission-on-marcc.md" %}
[slurm-submission-on-marcc.md](slurm-submission-on-marcc.md)
{% endcontent-ref %}

{% content-ref url="running-on-aws.md" %}
[running-on-aws.md](running-on-aws.md)
{% endcontent-ref %}

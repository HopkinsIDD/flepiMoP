---
description: >-
  Instructions on how to begin engaging with flepiMoP both remotely and locally.
  Provide users with a script that will locally install flepiMoP and all packages/dependencies.
---

# Before any run

## ⇅ Get set up to use Github

You need to interact with Github to run and edit `flepiMoP` code. [Github](https://github.com/) is a web platform for people to share and manage software, and it is based on a 'version control' software called `git` that helps programmers keep track of changes to code. Flepimop core code as well as example projects using flepimop code are all stored on Github, and frequently being updated. The first step to using flepimop for your own project is making sure you're set up to interact with code shared on Github.

If you are totally new to Github, navigate to [Github.com](https://github.com/) and Sign Up for a new account. Read about the [basics of git](https://docs.github.com/en/get-started/getting-started-with-git/set-up-git).

To work with `flepimop` code, you can do some tasks from the Github website, but you'll also need a way to 'clone' the code to your own local computer and keep it up to date with versions hosted online. You can do this either using a user interface like [Github Desktop](https://desktop.github.com/), or, using [`git` ](https://git-scm.com/downloads)commands from the command line. Make sure you have one or both installed.

If you are a veteran user, make sure you're signed in on Github.com and through whatever method you use locally on your computer to interact with Github.

## 🔐 Access the flepiMoP model code

In order to run a model with flepiMoP, you will need to clone the flepiMoP **code** to your machine. 

**To clone the `flepiMoP` code repository:**

* If you're using the command line in a terminal, first navigate to the local directory you'll use as the directory for the files that make up `flepiMoP`. Then, use the command: `git clone https://github.com/HopkinsIDD/flepiMoP`
* If you're using Github Desktop, go File -> Clone Repository, switch to the "URL" tab and copy the URL `https://github.com/HopkinsIDD/flepiMoP` there. For the "Local Path" option, make sure you choose your desired directory.

You can routinely ensure that your local clone of the flepiMoP code is up to date with upstream `flepiMoP` by navigating in terminal to your `flepiMoP` directory and using the command: `git pull`

# Locally install `flepiMoP`

First, ensure you have Python and R installed. You need a working Python.7+ installation. We recommend using the latest stable Python release (Python 3.12) to benefit from huge speed-ups and future-proof your installation. We also recommend installing Rstudio to interact with the R code and explore your model outputs.

## 🐍 Install `conda`

In order to complete `flepiMoP` installation, you must have [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) installed on your machine. `conda` is a tool that will assist you in managing software environments and code packages on your device, and it will be very helpful in ensuring consistent, reproducible environments across different projects. To install `conda` follow [the directions](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) according to your operating system. We would recommend selecting the `Anaconda Distribution` installer of `conda`.

Installation of `conda` may take a few minutes.

## ⬇️ Install flepiMoP packages and dependencies

Navigate to the `flepiMoP` directory and run the following command:

**Note: This installation script is currently only designed for Mac/Linux operating systems. Windows installation script coming soon.**
```bash
./build/local_install_or_update
```
Successfully running this script will do the following:
1. Determine `$FLEPI_PATH` and `$FLEPI_CONDA` environment variables
2. Activate your conda environment
3. Install `gempyor` and related Python dependencies
4. Install necessary R packages and dependencies

{% hint style="success" %}
Now you area ready to run the code using your desired method (see below)!
{% endhint %}
{% hint style="warning" %}
If you have any trouble or questions while trying to run `flepimop`, please report them on the [GitHub Q\&A](https://github.com/HopkinsIDD/flepiMoP/discussions/categories/q-a).
{% endhint %}

## 🤔 Deciding how to run

The code is written in a combination of [R](https://www.r-project.org/) and [Python](https://www.python.org/). The Python part of the model is a package called [_gempyor_](../gempyor/model-description.md), and includes all the code to simulate the epidemic model and the observational model and apply time-dependent interventions. The R component conducts the (optional) parameter inference, and all the (optional) provided pre and post processing scripts are also written in R. Most uses of the code require interacting with components written in both languages, and thus making sure that both are installed along with a set of required packages. However, Python alone can be used to do forward simulations of the model using `gempyor`.

Because of the need for multiple software packages and dependencies, we describe different ways you can run the model, depending on the requirements of your model setup. See [Quick Start Guide](quick-start-guide.md) for a quick introduction to using `gempyor` and `flepiMoP`. We also provide some more [advanced](advanced-run-guides/) ways to run our model, particularly for doing more complex model inference tasks.




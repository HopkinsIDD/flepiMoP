---
description: >-
    Instructions on how to proceed with the installations necessary for local flepiMoP use.
---


# Local flepiMoP installation

## ⇅ Get set up to use Github

You need to interact with Github to run and edit `flepiMoP` code. [Github](https://github.com/) is a web platform for people to share and manage software, and it is based on a 'version control' software called `git` that helps programmers keep track of changes to code. Flepimop core code as well as example projects using flepimop code are all stored on Github, and frequently being updated. The first step to using flepimop for your own project is making sure you're set up to interact with code shared on Github.

If you are totally new to Github, navigate to [Github.com](https://github.com/) and Sign Up for a new account. Read about the [basics of git](https://docs.github.com/en/get-started/getting-started-with-git/set-up-git).

To work with `flepimop` code, you can do some tasks from the Github website, but you'll also need a way to 'clone' the code to your own local computer and keep it up to date with versions hosted online. You can do this either using a user interface like [Github Desktop](https://desktop.github.com/), or, using [`git` ](https://git-scm.com/downloads)commands from the command line. Make sure you have one or both installed.

If you are a veteran user, make sure you're signed in on Github.com and through whatever method you use locally on your computer to interact with Github.

## 🔐 Access the flepiMoP model code

In order to run a model with flepiMoP, you will need to clone the flepiMoP **code** to your machine. 

**To clone the `flepiMoP` code repository:**

* If you're using the command line in a terminal, first navigate to the local directory you'll use as the directory for the files that make up `flepiMoP`. Then, use the command:\
  `git clone https://github.com/HopkinsIDD/flepiMoP`
* If you're using Github Desktop, go File -> Clone Repository, switch to the "URL" tab and copy the URL `https://github.com/HopkinsIDD/flepiMoP` there. For the "Local Path" option, make sure you choose your desired directory.

## 🐍 Installing `conda`

In order to complete `flepiMoP` installation, you must have [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) installed on your machine. `conda` is a tool that will assist you in managing software environments and code packages on your device. To install `conda` follow the directions according to your operating system:

* [Windows](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html)
* [macOS](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html)
* [Linux](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

We would recommend selecting the `Anaconda Distribution` installer of `conda`.

Once installed, you must set up your `conda` environmnet. To do this, navigate to your `flepiMoP` directory, and run the following commands:

```bash
conda env create -f /build/renv/conda_environment.yml
```
This will create a `conda` environment called `flepimop-env`, that you will activate with the command:

```bash
conda activate flepimop-env
```

## ⬇️ Installing flepiMoP packages and dependencies

While in the `flepiMoP` directory, run the following command:

```bash
/build/local_install_or_update.sh
```

This command will install the flepiMoP Python package, `gempyor`, along with all its dependencies, as well as all necessary R packages and dependencies. This command will also verify that your `conda` environment settings are corretly configured for flepiMoP use.
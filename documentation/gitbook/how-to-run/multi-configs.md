---
description: >-
  How to use multiple configuration files.
---

# Using Multiple Configuration Files

## 🧱 Set up

Create a sample project by copying from the examples folder:

```bash
mkdir myflepimopexample # or wherever
cd myflepimopexample
cp -r $FLEPI_PATH/examples/tutorials/* .
ls
```

You should see an assortment of yml files as a result of that `ls` command.

## Usage

If you run

```bash
flepimop simulate config_sample_2pop.yml
```

you'll get a basic foward simulation of this example model. However, you might also note there are several `*_part.yml` files, corresponding to partial configs. You can `simulate` using the combination of multiple configs with, for example:

```bash
flepimop simulate config_sample_2pop.yml config_sample_2pop_outcomes_part.yml
```

if want to see what the combined configuration is, you can use the `patch` command:

```bash
flepimop patch config_sample_2pop.yml config_sample_2pop_outcomes_part.yml
```

You may provide an arbitrary number of separate configuration files to combine to create a complete configuration.

At this time, only `simulate` supports multiple configuration files. Also, the patching operation is fairly crude: configuration options override previous ones completely, though with a warning. The files provided from left to right are from lowest priority (i.e. for the first file, only options specified in no other files are used) to highest priority (i.e. for the last file, its options override any other specification).

We are expanding coverage of this capability to other flepimop actions, e.g. inference, and are exploring options for smarter patching.
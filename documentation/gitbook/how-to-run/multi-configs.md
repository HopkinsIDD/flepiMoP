---
description: >-
  Patching together multiple configuration files.
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

if want to create a combined configuration (e.g. to check what the combined configuration is), you can use the `patch` command:

```bash
flepimop patch config_sample_2pop.yml config_sample_2pop_outcomes_part.yml > config_new.yml
cat config_new.yml
```

You may provide an arbitrary number of separate configuration files to combine to create a complete configuration.

## Caveats

At this time, only `simulate` supports multiple configuration files. Also, the patching operation is fairly crude: if multiple configuration files specify the same option, this will yield an error.

We are expanding coverage of this capability to other flepimop actions, e.g. inference, and are exploring options for smarter patching.

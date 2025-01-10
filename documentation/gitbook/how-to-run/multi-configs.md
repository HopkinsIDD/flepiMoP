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

If you run:

```bash
flepimop simulate config_sample_2pop.yml
```

You'll get a basic forward simulation of this example model. However, you might also note there are several `*_part.yml` files, corresponding to partial configs. You can `simulate` using the combination of multiple configs with, for example:

```bash
flepimop simulate config_sample_2pop.yml config_sample_2pop_outcomes_part.yml
```

While simulate can run your patched configuration, we also suggest you check your configuration file using the patch command:

```bash
flepimop patch config_sample_2pop.yml config_sample_2pop_outcomes_part.yml > config_new.yml
cat config_new.yml
```

You may provide an arbitrary number of separate configuration files to combine to create a complete configuration.

## Caveats

At this time, only simulate directly supports multiple configuration files, and our current patching capabilities only allow for the addition of new sections as given in our tutorials. This is helpful for building models piece-by-piece from a simple compartmental forward simulation, to including outcome probabilities, and finally, adding modifier sections. If multiple configuration files specify the same higher level configuration chunks (e.g., seir, outcomes), this will yield an error.

We are expanding coverage of this capability to other flepimop actions, e.g. inference, and are exploring options for smarter patching.

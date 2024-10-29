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

if want to see what the combined configuration is, you can use the `patch` command:

```bash
flepimop patch config_sample_2pop.yml config_sample_2pop_outcomes_part.yml
```

You may provide an arbitrary number of separate configuration files to combine to create a complete configuration.

## Caveats

At this time, only `simulate` supports multiple configuration files. Also, the patching operation is fairly crude: configuration options override previous ones completely, though with a warning. The files provided from left to right are from lowest priority (i.e. for the first file, only options specified in no other files are used) to highest priority (i.e. for the last file, its options override any other specification).

We are expanding coverage of this capability to other flepimop actions, e.g. inference, and are exploring options for smarter patching.

However, currently there are pitfalls like

```yaml
# config1
seir_modifiers:
  scenarios: ["one", "two"]
  one:
    # ...
  two:
    # ...
```

```yaml
# config2
seir_modifiers:
  scenarios: ["one", "three"]
  one:
    # ...
  three:
    # ...
```

Then you might expect

```bash
flepimop simulate config1.yml config2.yml
```

...to override seir scenario one and add scenario three, but what actually happens is that the entire seir_modifiers from config1 is overriden by config2. Specifying the configuration files in the reverse order would lead to a different outcome (the config1 seir_modifiers overrides config2 settings). If you're doing complex combinations of configuration files, you should use `flepimop patch ...` to ensure you're getting what you expect.
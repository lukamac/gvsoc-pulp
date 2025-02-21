# Questions

1. Exact internal definition of operations and bitwidths for scaling/accumulation/whatever:
    - I assumed now that scaling (multiplication) is done with 48bit saturating result
    - How is bias done?
    - How is shifting done?

1.1. I checked in latest RTL on github
    - int32 scaling is not supported
    - intermediate scaled bitwidth is of size acc_bitwidth + scale_bitwidth = 32 + 8 = 40 (which makes total sense)
    - how is it done in case of int32 scale prev? do we even want to support it? there is the question of quantization...

# Configurable Neureka

## Solutions

### JSON

The `--target-opt` solution.
You can change the configuration options in json like this:
```
gvsoc --target-opt chip/cluster/neureka/nid=4
```

Pros:
- already implemented cli interface (easily extandable)
- faster implementation

Cons:
- slower execution (less known stuff at compile-time)

### Templating

Pros:
- faster execution

Cons:
- slower compilation
    - have to compile each time I change the configuration...

## How to decide?
- best would be to do performance testing now when it's templated-like and see where is most time spent
- implement a few things from config (some widths e.g. or maybe PE sizing) and see change in perf

- I will have to do a lot of experiments with hopefully big netowrks...

## GVSoC support for template params
- check with germain
- that would be kinda hot
- or comptime json loading, that would be sexy

MVP:

Design documents describing this tool / feature set

It can run graphs

Needs to validate

Needs to time the runs

Engine selection

Needs to generate output in some form

Need to add a reference plugin for validation

Maybe TestSDK can generate a reference plugin

It’ll need to handle the device → host transfer since plugin API expects device buffers

Really validation should be just a baseline vrs another engine Id (and then we can set the Id to be a specific one)

Can support A/B tests by using the extension API, and allowing the user to specify a plugin path, or plugin name when describing the A vrs B.

Still need to supply the engineId as well, but can be like:

--APath “blah” --AId 12 --BPath”blahv2” --BId 12

Can make APath & BPath optional

Need A/B tests for validation, but not performance to start with.

Doesn’t need to be part of TheRock (yet), and can be a standalone application similar to samples where you need a full hipDNN install + whatever plugins you want to check.





hipDNN needs a full integration project that enabled E2E testing and performance capturing on the entire hipDNN install (core + plugins).



Looking to have a benchmarking project with the following support:

We want a python application that can be used for benchmarking and running hipDNN operation graphs & tracking important metrics.

Validation:

Has options for outputting mismatched values

Possible support for different validation backends (CPU reference, Pytorch)

Metrics we want to capture:

E2E runtime (with graph init)

E2E graph execution (minus graph init)

Performance details:

Easy to extract traces

Any extra details we can pull out (hardware counters, tflops, etc)

We want to be able to extract a full stats picture of the runs

MIOpenDriver for example will output stats that have been averaged, and in this case we want to be able to control the output information more detailed:

We want to capture each run values.

We want to be able to summarize the runs into some common stats (mean, STD, min, max, average, median etc)

We want to be able to run this given a graph

We want to be able to run this given a graph + tensors (like golden ref data tests)

We need some way to support bulk running graphs (possibly provide a folder instead of a file)

Support selecting specific backend as part of running the tool:

(Can specify which plugin to use)

Look into being able to support running with the same plugin loaded twice with different versions.

Might need some options here.

Investigate leveraging Pytorch & Triton for additional capabilities:

Pytorch can be used for validation

Triton already has some ability to capture performance stats

We want to be able to output stats in a few different formats:

CSV

Json

Supports existing operations

BN, Conv, + Activs

Ideally we want to be able to compare A/B runs (this will be a separate comparison tool)

Output stats from a specified run compared to a different run

Generate delta information / highlights

Might be a separate tool that can take outputs and generate visualizations / details.


This is the start of this project.
We will need to trim this scope down to a MVP version that is doable this milestone.


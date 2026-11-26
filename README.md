# Distributed Neural Network Training

**HPPC 2026 — Project 5**

## Project overview

This project studies how data-parallel training performs as the number of workers
increases. Each worker trains on a portion of the data, computes gradients, and
averages them across workers. We will implement gradient averaging with
`mpi4py` and MPI `Allreduce`, then compare it with PyTorch
`DistributedDataParallel` (DDP).

The main questions are:

- How does training throughput change as workers are added?
- How closely do the parallel implementations match a serial baseline?
- At what point does gradient communication start to limit performance?

## Dataset and model

The initial dataset will be **MNIST**, loaded through `torchvision`. We will use
the same model, preprocessing, optimizer, batch-size conventions, and training
workload when comparing implementations. CIFAR-10 may be used as a larger
follow-up workload if time and compute resources allow.

## Implementations

1. **Serial baseline:** Train on one worker and record training time, throughput,
   and validation accuracy.
2. **MPI gradient averaging:** Use `mpi4py` to distribute minibatches, compute
   local gradients with PyTorch, and average gradients with MPI `Allreduce`
   before each optimizer step.
3. **PyTorch DDP:** Train with `DistributedDataParallel` and compare against
   the custom MPI implementation. The target GPU comparison is two GPUs, as
   available on Kaggle's 2 × T4 runtime.

## Correctness

We will check the distributed implementations against the serial baseline using
a small, deterministic test. With matching initial weights and inputs, compare
the averaged distributed gradients with the serial gradients within a numerical
tolerance. We will also check that training completes and report validation
accuracy for each benchmark configuration.

## Performance experiments

We will run on at least two platforms: a laptop and Kaggle or Google Colab.
Available hardware and worker counts will be recorded for every run.

- **Strong scaling:** Keep the total training workload fixed and increase the
  number of workers.
- **Weak scaling:** Keep the workload per worker fixed as the worker count
  increases.
- Record elapsed time, samples per second, worker count, validation accuracy,
  and communication time or fraction where measurable.
- Write measurements from the benchmark program to CSV files so the results
  can be reproduced and plotted.

## Reproducibility

The completed repository will include the training and benchmark code, a
`Dockerfile` or `environment.yml`, and an automatically running correctness
test. Setup and run commands will be documented here alongside the finished
implementation. Benchmark results will include the platform, hardware,
software versions, dataset settings, and worker configuration.

No performance results are reported yet; they will be added after the
implementations and experiments are complete.

## Project requirements

The project deliverables include a one-page proposal, serial baseline and
correctness check, parallel implementations, a scaling study with measurements
from two platforms, reproducible repository setup and an automatic test, and a
10-minute presentation of at most 8 slides. The presentation will include a
flowchart of the project pipeline.

See [HPPC Project Topics.pdf](./HPPC%20Project%20Topics.pdf) for the original
topic description.

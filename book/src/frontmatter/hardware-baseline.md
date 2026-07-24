# The hardware baseline

Every measured number in this book is relative to one machine. If I tell you a serving configuration hits some throughput, or a training step takes some seconds, or a KV cache eats some fraction of VRAM, that number only means something once you know exactly what it ran on. So before any measurement appears, here is the machine it was measured on, pinned down to the parts that matter.

The baseline machine is an MSI Aegis R2, model A2NVV9-2218US, a prebuilt desktop I chose specifically because it is an ordinary, buyable configuration rather than a bespoke workstation. The whole premise of the book is that the loop runs on hardware a normal person can own, so the baseline had to be hardware a normal person can buy.

## The full specification

| Component | Part |
|---|---|
| Machine | MSI Aegis R2 (A2NVV9-2218US) |
| CPU | Intel Core Ultra 9 285 (Arrow Lake) |
| GPU | NVIDIA GeForce RTX 5080, 16GB GDDR7 (Blackwell), roughly 960 GB/s memory bandwidth |
| System RAM | 32GB DDR5 |
| Working storage | 2TB SSD (1TB NVMe + 1TB SATA SSD moved over from the NAS) |
| Archive storage | 4TB HDD NAS (network-attached), fronted by MinIO |
| OS | Ubuntu 24.04 LTS |
| GPU driver | NVIDIA 570-open (open-kernel-module branch) |

Whenever I refer to "the baseline machine" anywhere in the book, this is the row-by-row thing I mean.

## Why each part matters

The GPU is the whole story, and two of its numbers set the terms for everything else. It is a Blackwell-generation card, which matters because the software stack (the CUDA version, the driver branch, the kernels that vLLM and the training libraries dispatch to) has to actually support Blackwell, and at the time I built this that support was new enough to be worth stating explicitly rather than assuming. And it has 16GB of VRAM, which is the constraint the entire book is organized around. Every `vram-budget` admonition is accounting against this 16GB ceiling. The roughly 960 GB/s of memory bandwidth is the number that quietly governs inference speed for the memory-bound regime we usually live in: for autoregressive decoding, throughput is often set more by how fast you can stream weights and KV cache through memory than by raw compute, so the bandwidth figure is one I will keep returning to when a measurement looks surprising.

The CPU is an Intel Core Ultra 9 285, an Arrow Lake part. It is not doing the heavy lifting, but it is not idle either. Data loading, tokenization, the orchestration around a serving engine, and the CPU-side of any offloading strategy all land here, and when a lab is unexpectedly slow it is worth knowing whether the GPU is waiting on the CPU. Arrow Lake also brings its own quirks (its power and scheduling behavior differs from older Intel parts), which is exactly the kind of thing that is invisible until it explains a weird measurement.

System RAM is 32GB of DDR5. That is comfortable but not lavish, and it becomes relevant the moment a technique spills off the GPU: CPU offload of optimizer state or KV cache, memory-mapped weight loading, or just holding a dataset in memory. 32GB means offloading is possible but bounded, and I will flag the cases where it is the thing that fits or does not.

Storage is deliberately split into two tiers, and the split is a design choice, not an accident of what was in the box.

## Working storage versus archive

The 2TB of local SSD is working storage: a 1TB NVMe drive plus a 1TB SATA SSD I pulled out of the NAS and moved into the machine, precisely because the machine is where fast disk earns its keep. It holds the things a running lab touches constantly: the model weights currently being served or trained, the active dataset, checkpoints being written mid-run, the scratch space for a training job, and the hot copies of any data index a query path hits. Local SSD is here because weight loading and checkpoint I/O are fast enough to not be the bottleneck, which matters when a lab loads a multi-gigabyte model or writes a checkpoint every few hundred steps.

The 4TB HDD NAS is the archive tier, and it wears an S3-compatible face: a single MinIO instance runs on the NAS, so everything durable is addressable as `s3://` by every tool in the stack. It holds the things I want to keep but am not actively hammering: past checkpoints, completed run artifacts, raw data snapshots, datasets I am not currently using, the accumulated sediment of every lab I have ever run. It is network-attached spinning disk and therefore slower than local SSD, which is fine, because nothing in the hot path should be reaching across the network for it. The discipline is to keep the working set on local SSD and everything else on the NAS, so the fast disks stay fast and the archive grows without pressure.

That is the high-level shape. The actual mechanics (how the mount is set up, how artifacts get promoted from working to archive, how I keep the two from drifting) are a chapter 0.5 problem, and I will lay out the whole storage workflow there. For now the thing to hold onto is just the two-tier split: NVMe for what is hot, NAS for what is kept.

## Dating and driver-stamping every number

Here is the discipline that makes this baseline useful rather than decorative. Every genuinely measured number in this book is recorded with the date it was measured and the driver it was measured under. Not "about 40 tokens per second" but "40 tokens per second, measured on the baseline machine on such-and-such date under driver 570-open." Where a number in the text is a real measurement, you will see it written as `(measured on the baseline machine — record value, date, driver)` until it is filled in from an actual run, precisely so that an unfilled measurement is impossible to mistake for a real one.

The reason is that hardware and drivers move, and a number without a stamp is a number you cannot trust later. A driver update can change throughput. A new CUDA release can change what fits in memory. A kernel improvement can make last month's careful workaround unnecessary. If every measurement carries its date and driver, then when I upgrade the driver next year I can re-run the labs and diff the new numbers against the old ones and actually see what the upgrade changed. If the numbers were undated, that diff would be meaningless, because I would not know whether a difference was the upgrade or just drift I never pinned down.

This is the same promise I made to future me in the preface, made concrete. A dated, driver-stamped measurement is reproducible in the only sense that matters: I can tell you the conditions precisely enough that you, or a later me, can reproduce them and check. An undated number is a rumor. So the rule for the whole book is that measured quantities are never bare. They come stamped, or they come marked as not yet measured, and there is no third category.

With the machine pinned and the stamping discipline stated, every measurement from here on has a fixed frame of reference. The next chapter starts the actual work: unboxing this machine and getting it ready to serve its first model.

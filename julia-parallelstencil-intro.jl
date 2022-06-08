#src # This is needed to make this run as normal Julia file
using Markdown #src

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
# Parallel high-performance stencil computations on xPUs

Ludovic Räss

_ETH Zurich_

"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""

_supported by_

Sam Omlin, Ivan Utkin, Mauro Werder

"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## Why to solve PDEs on xPUs ... or GPUs

![gpu](./figures/gpu.png)

"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
### A brief intro about GPU computing:
- Why we do it
- Why it is cool (in Julia)
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
### Why we do it
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
Predict the evolution of natural and engineered systems
- e.g. ice cap evolution, stress distribution, etc...

![ice2](./figures/ice2.png)
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
Physical processes that describe those systems are **complex** and often **nonlinear**
- no or very limited analytical solution is available

👉 a numerical approach is required to solve the mathematical model
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
Solving PDEs is computationally demanding
- ODEs - scalar equations

$$ \frac{∂C}{∂t} = -\frac{(C-C_{eq})}{ξ} $$

but...
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
- PDEs - involve vectors (and tensors)  👉 local gradients & neighbours

$$ \frac{∂C}{∂t} = D~ \left(\frac{∂^2C}{∂x^2} + \frac{∂^2C}{∂y^2} \right) $$

"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
Computational costs increase
- with complexity (e.g. multi-physics, couplings)
- with dimensions (3D tensors...)
- upon refining spatial and temporal resolution

"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
![Stokes2D_vep](./figures/Stokes2D_vep.gif)

"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
Use **parallel computing** (to address this):
- The "memory wall" in ~ 2004
- Single-core to multi-core devices

![mem_wall](./figures/mem_wall.png)

"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
GPUs are massively parallel devices
- SIMD machine (programmed using threads - SPMD) ([more](https://safari.ethz.ch/architecture/fall2020/lib/exe/fetch.php?media=onur-comparch-fall2020-lecture24-simdandgpu-afterlecture.pdf))
- Further increases the Flop vs Bytes gap

![cpu_gpu_evo](./figures/cpu_gpu_evo.png)

"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
Taking a look at a recent GPU and CPU:
- Nvidia Tesla A100 GPU
- AMD EPYC "Rome" 7282 (16 cores) CPU

| Device         | TFLOP/s (FP64) | Memory BW TB/s |
| :------------: | :------------: | :------------: |
| Tesla A100     | 9.7            | 1.55           |
| AMD EPYC 7282  | 0.7            | 0.085          |

"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
Current GPUs (and CPUs) can do many more computations in a given amount of time than they can access numbers from main memory.
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
Quantify the imbalance:

$$ \frac{\mathrm{computation\;peak\;performance\;[TFLOP/s]}}{\mathrm{memory\;access\;peak\;performance\;[TB/s]}} × \mathrm{size\;of\;a\;number\;[Bytes]} $$

"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
_(Theoretical peak performance values as specified by the vendors can be used)._
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
Back to our hardware:

| Device         | TFLOP/s (FP64) | Memory BW TB/s | Imbalance (FP64)     |
| :------------: | :------------: | :------------: | :------------------: |
| Tesla A100     | 9.7            | 1.55           | 9.7 / 1.55  × 8 = 50 |
| AMD EPYC 7282  | 0.7            | 0.085          | 0.7 / 0.085 × 8 = 66 |


_(here computed with double precision values)_
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
**Meaning:** we can do 50 (GPU) and 66 (CPU) floating point operations per number accessed from main memory. Floating point operations are "for free" when we work in memory-bounded regimes

👉 Requires to re-think the numerical implementation and solution strategies
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
### On the scientific application side

- Most algorithms require only a few operations or flops ...
- ... compared to the amount of numbers or bytes accessed from main memory.
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
First derivative example $∂A / ∂x$:
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
If we "naively" compare the "cost" of an isolated evaluation of a finite-difference first derivative, e.g., computing a flux $q$:

$$q = -D~\frac{∂A}{∂x}~,$$

which in the discrete form reads `q[ix] = -D*(A[ix+1]-A[ix])/dx`.
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
The cost of evaluating `q[ix] = -D*(A[ix+1]-A[ix])/dx`:

1 reads + 1 write => $2 × 8$ = **16 Bytes transferred**

1 (fused) addition and division => **1 floating point operations**
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
assuming:
- $D$, $∂x$ are scalars
- $q$ and $A$ are arrays of `Float64` (read from main memory)
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
GPUs and CPUs perform 50 - 60 FLOP pro number accessed from main memory

First derivative evaluation requires to transfer 2 numbers per FLOP
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
The FLOP/s metric is no longer the most adequate for reporting the application performance of many modern applications on modern hardware.
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
### Effective memory throughput metric $T_\mathrm{eff}$

Need for a memory throughput-based performance evaluation metric: $T_\mathrm{eff}$ [GB/s]

➡ Evaluate the performance of iterative stencil-based solvers.
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
The effective memory access $A_\mathrm{eff}$ [GB]

Sum of:
- twice the memory footprint of the unknown fields, $D_\mathrm{u}$, (fields that depend on their own history and that need to be updated every iteration)
- known fields, $D_\mathrm{k}$, that do not change every iteration. 
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
The effective memory access divided by the execution time per iteration, $t_\mathrm{it}$ [sec], defines the effective memory throughput, $T_\mathrm{eff}$ [GB/s]:

$$ A_\mathrm{eff} = 2~D_\mathrm{u} + D_\mathrm{k} $$

$$ T_\mathrm{eff} = \frac{A_\mathrm{eff}}{t_\mathrm{it}} $$
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
The upper bound of $T_\mathrm{eff}$ is $T_\mathrm{peak}$ as measured, e.g., by [McCalpin, 1995](https://www.researchgate.net/publication/51992086_Memory_bandwidth_and_machine_balance_in_high_performance_computers) for CPUs or a GPU analogue. 

Defining the $T_\mathrm{eff}$ metric, we assume that:
1. we evaluate an iterative stencil-based solver,
2. the problem size is much larger than the cache sizes and
3. the usage of time blocking is not feasible or advantageous (reasonable for real-world applications).
"""

#nb # > 💡 note: Fields within the effective memory access that do not depend on their own history; such fields can be re-computed on the fly or stored on-chip.

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
### Why it is cool
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
![julia-gpu](./figures/julia-gpu.png)

"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
#### GPU are cool
Price vs Performance
- Close to **1TB/s** memory throughput (here on nonlinear diffusion SIA)

![perf_gpu](./figures/perf_gpu.png)

_And one can get there_

"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
Availability (less fight for resources)
- Still not many applications run on GPUs

Workstation turns into a personal Supercomputers
- GPU vs CPUs peak memory bandwidth: theoretical 10x (practically maybe more)

![titan_node](./figures/titan_node.jpg)

"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
#### Julia is cool
Solution to the "two-language problem"

![two_lang](./figures/two_lang.png)

Single code for prototyping and production

"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
Backend agnostic:
- Single code to run on single CPU or thousands of GPUs
- Single code to run on various CPUs (x86, ARM, Power9, ...) \
  and GPUs (Nvidia, AMD, Intel?)

Interactive:
- No need for third-party visualisation software
- Debugging and interactive REPL mode
- Efficient for development
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
too good to be true?
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
![ParallelStencil](./figures/parallelstencil.png)
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
[https://github.com/omlins/ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl)
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## Let's get started with a concise demo using [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl)
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
We will now continue in the notebook you can freely fetch on GitHub at [https://github.com/luraess/julia-day-2022](https://github.com/luraess/julia-day-2022)
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
In the notebook, activate the environment:
"""
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()



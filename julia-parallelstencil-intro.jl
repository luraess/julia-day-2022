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

_dev team_

_Sam Omlin, Ivan Utkin, Mauro Werder_
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## The nice to have features

Wouldn't it be nice to have single code that:
- runs both on CPUs and GPUs (xPUs)?
- one can use for prototyping and production?
- runs at optimal performance?
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
Hold on 🙂 ...
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## Why to bother with GPU computing

A brief intro about GPU computing:
- Why we do GPU computing
- Why the Julia choice

![gpu](./figures/gpu.png)
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
### Why we do GPU computing
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

$$ \frac{\mathrm{computation\;peak\;perf.\;[TFLOP/s]}}{\mathrm{memory\;access\;peak\;perf.\;[TB/s]}} × \mathrm{size\;of\;a\;number\;[Bytes]} $$

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
**Meaning:** we can do about 50 floating point operations per number accessed from main memory. Floating point operations are "for free" when we work in memory-bounded regimes.

👉 Requires to re-think the numerical implementation and solution strategies
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
Unfortunately, the cost of evaluating a first derivative $∂A / ∂x$ using finite-differences

`q[ix] = -D*(A[ix+1]-A[ix])/dx`

consists of:
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
1 reads + 1 write => $2 × 8$ = **16 Bytes transferred**

1 (fused) addition and division => **1 floating point operations**

> assuming $D$, $∂x$ are scalars, $q$ and $A$ are arrays of `Float64` (read from main memory)
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## How to evaluate performance

_The FLOP/s metric is no longer the most adequate for reporting the application performance of many modern applications on modern hardware._
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
### Effective memory throughput metric $T_\mathrm{eff}$

Need for a memory throughput-based performance evaluation metric: $T_\mathrm{eff}$ [GiB/s]

➡  Evaluate the performance of iterative stencil-based solvers.
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
The effective memory access $A_\mathrm{eff}$ [GiB], is the sum of:
- twice the memory footprint of the unknown fields, $D_\mathrm{u}$, (fields that depend on their own history and that need to be updated every iteration)
- known fields, $D_\mathrm{k}$, that do not change every iteration. 
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
The effective memory access divided by the execution time per iteration, $t_\mathrm{it}$ [sec], defines the effective memory throughput, $T_\mathrm{eff}$ [GiB/s]:

$$ A_\mathrm{eff} = 2~D_\mathrm{u} + D_\mathrm{k}, \;\;\; T_\mathrm{eff} = \frac{A_\mathrm{eff}}{t_\mathrm{it}} $$

"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
The upper bound of $T_\mathrm{eff}$ is $T_\mathrm{peak}$ as measured, e.g., by [McCalpin, 1995](https://www.researchgate.net/publication/51992086_Memory_bandwidth_and_machine_balance_in_high_performance_computers) for CPUs or a GPU analogue. 

Defining the $T_\mathrm{eff}$ metric, we assume that:
1. we evaluate an iterative stencil-based solver,
2. the problem size is much larger than the cache sizes and
3. the usage of time blocking is not feasible or advantageous (reasonable for real-world applications).

> 💡 note: All "convenience" fields (that do not depend on their own history) should not be stored and can be re-computed on the fly or stored on-chip.
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
### Why the Julia choice
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
Julia + GPUs  ➡  close to **1 TB/s** memory throughput

![perf_gpu](./figures/perf_gpu.png)

**And one can get there** 🚀
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
#### Solution to the "two-language problem"

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


[https://github.com/omlins/ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl)
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## Enough propaganda
Let's check out [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl)
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
We'll solve the heat diffusion equation

$$ c \frac{∂T}{∂t} = ∇⋅λ ∇T $$

using explicit 2D finite-differences on a Cartesian staggered grid
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
👉 This notebook is available on GitHub at [https://github.com/luraess/julia-day-2022](https://github.com/luraess/julia-day-2022)
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## Heat solver implementations and performance evaluation

1. Array programming and broadcasting (vectorised Julia CPU)
2. Array programming and broadcasting (vectorised Julia GPU)
3. Kernel programming using ParallelStencil with math-close notation (`FiniteDifferences` module)
4. Advanced kernel programming using ParallelStencil (`@parallel_indices` features)
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## Setting up the environment

Before we start, let's activate the environment:
"""
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.status()

md"""
And add the package(s) we will use
"""
using Plots, CUDA, BenchmarkTools

md"""
### 1. Array programming on CPU

$$ c \frac{∂T}{∂t} = ∇⋅λ ∇T $$

A 24 lines code including visualisation:
"""
function diffusion2D()
    ## Physics
    λ      = 1.0                                           # Thermal conductivity
    c0     = 1.0                                           # Heat capacity
    lx, ly = 10.0, 10.0                                    # Length of computational domain in dimension x and y
    ## Numerics
    nx, ny = 32*2, 32*2                                    # Number of grid points in dimensions x and y
    nt     = 100                                           # Number of time steps
    dx, dy = lx/(nx-1), ly/(ny-1)                          # Space step in x and y-dimension
    ## Array initializations
    T      = zeros(Float64,nx  ,ny  )                      # Temperature
    Ci     = zeros(Float64,nx  ,ny  )                      # 1/Heat capacity
    qTx    = zeros(Float64,nx-1,ny-2)                      # Heat flux in x-dim
    qTy    = zeros(Float64,nx-2,ny-1)                      # Heat flux in y-dim
    ## Initial conditions
    Ci    .= 1.0/c0                                        # 1/Heat capacity (could vary in space)
    T     .= [exp(-(((ix-1)*dx-lx/2)/2)^2-(((iy-1)*dy-ly/2)/2)^2) for ix=1:size(T,1), iy=1:size(T,2)] # Initial Gaussian Temp
    ## Time loop
    dt     = min(dx^2,dy^2)/λ/maximum(Ci)/4.1              # Time step for 2D Heat diffusion
    opts   = (aspect_ratio=1,xlims=(1,nx),ylims=(1,ny),clims=(0.0,1.0),c=:turbo,xlabel="Lx",ylabel="Ly") # plotting options
    @gif for it = 1:nt
        qTx .= .-λ .* diff(T[:,2:end-1],dims=1)./dx
        qTy .= .-λ .* diff(T[2:end-1,:],dims=2)./dy
        T[2:end-1,2:end-1] .= T[2:end-1,2:end-1] .+ dt.*Ci[2:end-1,2:end-1].*(.-diff(qTx,dims=1)./dx .-diff(qTy,dims=2)./dy)
        heatmap(Array(T)',title="it=$it"; opts...)        # Visualization
    end
end
#-
diffusion2D()

md"""
The above example runs on the CPU. What if we want to execute it on the GPU?

### 2. Array programming on GPU

In Julia, this is pretty simple as we can use the [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) package 
"""
using CUDA

md"""
and add initialise our arrays as `CuArray`s:
"""
function diffusion2D()
    ## Physics
    λ      = 1.0                                           # Thermal conductivity
    c0     = 1.0                                           # Heat capacity
    lx, ly = 10.0, 10.0                                    # Length of computational domain in dimension x and y
    ## Numerics
    nx, ny = 32*2, 32*2                                    # Number of grid points in dimensions x and y
    nt     = 100                                           # Number of time steps
    dx, dy = lx/(nx-1), ly/(ny-1)                          # Space step in x and y-dimension
    ## Array initializations
    T      = CUDA.zeros(Float64,nx  ,ny  )                 # Temperature
    Ci     = CUDA.zeros(Float64,nx  ,ny  )                 # 1/Heat capacity
    qTx    = CUDA.zeros(Float64,nx-1,ny-2)                 # Heat flux in x-dim
    qTy    = CUDA.zeros(Float64,nx-2,ny-1)                 # Heat flux in y-dim
    ## Initial conditions
    Ci    .= 1.0/c0                                        # 1/Heat capacity (could vary in space)
    T     .= CuArray([exp(-(((ix-1)*dx-lx/2)/2)^2-(((iy-1)*dy-ly/2)/2)^2) for ix=1:size(T,1), iy=1:size(T,2)]) # Initial Gaussian Temp
    ## Time loop
    dt     = min(dx^2,dy^2)/λ/maximum(Ci)/4.1              # Time step for 2D Heat diffusion
    opts   = (aspect_ratio=1,xlims=(1,nx),ylims=(1,ny),clims=(0.0,1.0),c=:turbo,xlabel="Lx",ylabel="Ly") # plotting options
    @gif for it = 1:nt
        qTx .= .-λ .* diff(T[:,2:end-1],dims=1)./dx
        qTy .= .-λ .* diff(T[2:end-1,:],dims=2)./dy
        T[2:end-1,2:end-1] .= T[2:end-1,2:end-1] .+ dt.*Ci[2:end-1,2:end-1].*(.-diff(qTx,dims=1)./dx .-diff(qTy,dims=2)./dy)
        heatmap(Array(T)',title="it=$it"; opts...)        # Visualization
    end
end
#-
diffusion2D()

md"""
Nice, so it runs on the GPU now. But how much faster - what did we gain? Let's determine the effective memory throughput $T_\mathrm{eff}$ for both implementations.

### CPU vs GPU array programming performance

For this, we can isolate the physics computation into a function that we will evaluate for benchmarking
"""
function update_temperature!(T, qTx, qTy, Ci, λ, dt, dx, dy)
    @inbounds qTx .= .-λ .* diff(T[:,2:end-1],dims=1)./dx
    @inbounds qTy .= .-λ .* diff(T[2:end-1,:],dims=2)./dy
    @inbounds T[2:end-1,2:end-1] .= T[2:end-1,2:end-1] .+ dt.*Ci[2:end-1,2:end-1].*(.-diff(qTx,dims=1)./dx .-diff(qTy,dims=2)./dy)
    return
end

md"""
Moreover, for benchmarking activities, we will require the following arrays and scalars and make sure to use sufficiently large arrays in order to saturate the memory bandwidth:
"""
nx = ny = 512#*32
T   = rand(Float64,nx  ,ny  )
Ci  = rand(Float64,nx  ,ny  )
qTx = rand(Float64,nx-1,ny-2)
qTy = rand(Float64,nx-2,ny-1)
λ = dx = dy = dt = rand();

md"""
And use `@belapsed` macro from [BenchmarTools](https://github.com/JuliaCI/BenchmarkTools.jl) to sample our perf:
"""
t_it = @belapsed begin update_temperature!($T, $qTx, $qTy, $Ci, $λ, $dt, $dx, $dy); end
T_eff_cpu_bcast = (2*1+1)*1/1e9*nx*ny*sizeof(Float64)/t_it
println("T_eff = $(T_eff_cpu_bcast) GiB/s using CPU array programming")

md"""
Let's repeat the experiment using the GPU
"""
nx = ny = 512#*32
T   = CUDA.rand(Float64,nx  ,ny  )
Ci  = CUDA.rand(Float64,nx  ,ny  )
qTx = CUDA.rand(Float64,nx-1,ny-2)
qTy = CUDA.rand(Float64,nx-2,ny-1)
λ = dx = dy = dt = rand();

md"""
And sample again our performance from the GPU execution this time:
"""
t_it = @belapsed begin update_temperature!($T, $qTx, $qTy, $Ci, $λ, $dt, $dx, $dy); end
T_eff_gpu_bcast = (2*1+1)*1/1e9*nx*ny*sizeof(Float64)/t_it
println("T_eff = $(T_eff_gpu_bcast) GiB/s using GPU array programming")

md"""
We see some improvement from performing the computations on the GPU, however, $T_\mathrm{eff}$ is not yet close to GPU's peak memory bandwidth

How to improve? Now it's time for ParallelStencil

### 3. Kernel programming using ParallelStencil

In this first example, we'll use the `FiniteDifferences` module to enable math-close notation and the `CUDA` "backend". We could simply switch the backend to `Threads` if we want the same code to run on multiple CPU threads using Julia's native multi-threading capabilities. But for time issues, we won't investigate this today.
"""
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(Threads, Float64, 2)
## @init_parallel_stencil(CUDA, Float64, 2)
## CUDA.device!(7) # select specific GPU
nx = ny = 512#*32
T   = @rand(nx  ,ny  )
Ci  = @rand(nx  ,ny  )
qTx = @rand(nx-1,ny-2)
qTy = @rand(nx-2,ny-1)
λ = dx = dy = dt = rand();

md"""
Using math-close notations from the `FiniteDifferences2D` module, our update kernel can be re-written as following:
"""
@parallel function update_temperature_ps!(T, qTx, qTy, Ci, λ, dt, dx, dy)
    @all(qTx) = -λ * @d_xi(T)/dx
    @all(qTy) = -λ * @d_yi(T)/dy
    @inn(T)   = @inn(T) + dt*@inn(Ci)*(-@d_xa(qTx)/dx -@d_ya(qTy)/dy)
    return
end

md"""
And sample again our performance on the GPU using ParallelStencil this time:
"""
t_it = @belapsed begin @parallel update_temperature_ps!($T, $qTx, $qTy, $Ci, $λ, $dt, $dx, $dy); end
T_eff_ps = (2*1+1)*1/1e9*nx*ny*sizeof(Float64)/t_it
println("T_eff = $(T_eff_ps) GiB/s using ParallelStencil on GPU and the FiniteDifferences2D module")

md"""
It's already much better, but we can do more in order to approach the peak memory bandwidth of the GPU.

### 4. Advanced kernel programming using ParallelStencil (`@parallel_indices` features)

We should remove the convenience arrays `qTx`, `qTy` (no time dependence).
We then also need to switch from the `@parallel` to `@parallel_indices (ix,iy)` function declaration.
We can, e.g., use macros to still have our fluxes explicitly written down, resulting in:
"""
T2 = copy(T)
macro qTx(ix,iy)  esc(:( -λ*(T[$ix+1,$iy+1] - T[$ix,$iy+1])/dx )) end
macro qTy(ix,iy)  esc(:( -λ*(T[$ix+1,$iy+1] - T[$ix+1,$iy])/dy )) end
@parallel_indices (ix,iy) function update_temperature_psind!(T2, T, Ci, λ, dt, dx, dy)
    nx, ny = size(T2)
    if (ix>1 && ix<nx && iy>1 && iy<ny)
        @inbounds T2[ix+1,iy+1] = T[ix+1,iy+1] + dt*Ci[ix,iy]*( -(@qTx(ix+1,iy) - @qTx(ix,iy))/dx -(@qTy(ix,iy+1) - @qTy(ix,iy))/dy )
    end
    return
end

md"""
> 💡 note: We need a buffer array now in order to avoid race conditions and erroneous results when accessing the `T` array in parallel.

We can now sample again our performance on the GPU using `parallel_indices` this time:
"""
t_it = @belapsed begin @parallel update_temperature_psind!($T2, $T, $Ci, $λ, $dt, $dx, $dy); end
T_eff_psind = (2*1+1)*1/1e9*nx*ny*sizeof(Float64)/t_it
println("T_eff = $(T_eff_psind) GiB/s using ParallelStencil on GPU and @parallel_indices")

md"""
So, we made it. Time to recap what we've seen.

## Conclusions

- Starting with performance, we can now clearly see our 4 data points of $T_\mathrm{eff}$ and how close the GPU perf is from the peak memory bandwidth
"""
xPU  = ("CPU AP", "GPU AP", "GPU PS FD", "GPU PS perf")
Teff = (T_eff_cpu_bcast, T_eff_gpu_bcast, T_eff_ps, T_eff_psind)
plot(Teff,ylabel="T_eff",xlabel="implementation",xticks=(1:length(xPU),xPU),xaxis=([0.7, 4.3]),linewidth=0,markershape=:square,markersize=8,legend=false,fontfamily="Courier",framestyle=:box)

md"""
- Julia and ParallelStencil permit to solve the two-language problem
- ParallelStencil and Julia GPU permit to exploit close to GPUs' peak memory throughput

![parallelstencil](./figures/parallelstencil.png)

More hungry? Check out [https://github.com/omlins/ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) and the [miniapps](https://github.com/omlins/ParallelStencil.jl#concise-singlemulti-xpu-miniapps)

Advance features not covered today:
- Using shared-memory and 2.5D blocking (see [here](https://github.com/omlins/ParallelStencil.jl#support-for-architecture-agnostic-low-level-kernel-programming) with [example](https://github.com/omlins/ParallelStencil.jl/blob/main/examples/diffusion2D_shmem_novis.jl))
- Multi-GPU with communication-computation overlap combining ParallelStencil and [ImplicitGlobalGrid]()
- Stay tuned, AMDGPU support is coming soon 🚀

_contact: luraess@ethz.ch_
"""

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

And solve the 2D heat diffusion.
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
👉 We will now continue in the notebook you can freely access on GitHub at [https://github.com/luraess/julia-day-2022](https://github.com/luraess/julia-day-2022)
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## Setting up the environment

In the notebook, activate the environment:
"""
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.status()

md"""
And add the package(s) we will use
"""
using Plots, CUDA, Benchmarktools

md"""
## Solving the 2D heat diffusion

$$ c \frac{∂T}{∂t} = ∇⋅λ ∇T $$

Let's implement an explicit diffusion solver using finite-differences and array programming together with broadcasting in "plain" Julia:
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
    nvis   = 10
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
        if it%nvis==0 heatmap(Array(T)',title=it; opts...) end      # Visualization
    end
end
#-
diffusion2D()

md"""
The above example runs on CPU. What if we want to execute it on the GPU? In Julia, this is pretty simple as we can use the [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) package 
"""
using CUDA

md"""
and add the `CUDA` key to the array initialisation as following:
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
    nvis   = 10
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
        if it%nvis==0 heatmap(Array(T)',title=it; opts...) end      # Visualization
    end
end
#-
diffusion2D()

md"""
Nice, so it runs on the GPU now. But how much faster - what did we gain?

### CPU array programming performance

Let's determine the effective memory throughput $T_\mathrm{eff}$. For this, we can isolate the physics computation into a function that we will use for benchmarking
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
println("T_eff = $(T_eff_cpu_bcast) GiB/s")

md"""
### GPU array programming performance

Let's repeat the experiment using the GPU
"""
nx = ny = 512#*32
T   = CUDA.rand(Float64,nx  ,ny  )
Ci  = CUDA.rand(Float64,nx  ,ny  )
qTx = CUDA.rand(Float64,nx-1,ny-2)
qTy = CUDA.rand(Float64,nx-2,ny-1)
λ = dx = dy = dt = rand();

md"""
And sample again our perf on the GPU this time:
"""
t_it = @belapsed begin update_temperature!($T, $qTx, $qTy, $Ci, $λ, $dt, $dx, $dy); end
T_eff_gpu_bcast = (2*1+1)*1/1e9*nx*ny*sizeof(Float64)/t_it
println("T_eff = $(T_eff_gpu_bcast) GiB/s")

md"""
Some blabla about perf.

## Using ParallelStencil

Finite difference module and `CUDA` "backend".
"""
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(Threads, Float64, 2)
## @init_parallel_stencil(CUDA, Float64, 2)
nx = ny = 512#*32
T   = @rand(nx  ,ny  )
Ci  = @rand(nx  ,ny  )
qTx = @rand(nx-1,ny-2)
qTy = @rand(nx-2,ny-1)
λ = dx = dy = dt = rand();

md"""
Using math-close notations from the FD module:
"""
@parallel function update_temperature_ps!(T, qTx, qTy, Ci, λ, dt, dx, dy)
    @all(qTx) = -λ * @d_xi(T)/dx
    @all(qTy) = -λ * @d_yi(T)/dy
    @inn(T)   = @inn(T) + dt*@inn(Ci)*(-@d_xa(qTx)/dx -@d_ya(qTy)/dy)
    return
end

md"""
And sample again our perf on the GPU using ParallelStencil this time:
"""
t_it = @belapsed begin @parallel update_temperature_ps!($T, $qTx, $qTy, $Ci, $λ, $dt, $dx, $dy); end
T_eff_ps = (2*1+1)*1/1e9*nx*ny*sizeof(Float64)/t_it
println("T_eff = $(T_eff_ps) GiB/s")

md"""
It's better, but we can do more in order to approach the peak memory bandwidth of the GPU
Removing the convenience arrays `qTx`, `qTy`:
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
And sample again our perf on the GPU using `parallel_indices` this time:
"""
t_it = @belapsed begin @parallel update_temperature_psind!($T2, $T, $Ci, $λ, $dt, $dx, $dy); end
T_eff_psind = (2*1+1)*1/1e9*nx*ny*sizeof(Float64)/t_it
println("T_eff = $(T_eff_psind) GiB/s")





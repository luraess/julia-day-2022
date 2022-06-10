# Julia-day 2022
Intro to parallel stencil computation in Julia

Program: https://calcul.math.cnrs.fr/2022-06-journee-julia-calcul.html


## Automatic notebook generation

The presentation slides and the demo notebook are self-contained in a Jupyter notebook [julia-parallelstencil-intro.ipynb](julia-parallelstencil-intro.ipynb) that can be auto-generated using literate programming. To reproduce:

1. Clone this git repo
2. Open Julia and resolve/instantiate the project
```julia
using Pkg
Pkg.activate(@__DIR__)
Pkg.resolve()
Pkg.instantiate()
```
3. Run the deploy script
```julia-repl
julia> using Literate

julia> include("deploy_notebooks.jl")
```
4. Then using IJulia, you can launch the notebook and get it displayed in your web browser:
```julia-repl
julia> using IJulia

julia> notebook(dir="./")
```
_To view the notebook as slide, you need to install the [RISE](https://rise.readthedocs.io/en/stable/installation.html) plugin_

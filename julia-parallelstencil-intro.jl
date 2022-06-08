#src # This is needed to make this run as normal Julia file
using Markdown #src

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
# GPU computing and performance assessment
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
### The goal of this lecture 9 is to:

- learn how to use shared memory (on-chip) to avoid main memory accesses and communicate between threads; and
- learn how to control registers for storing intermediate results on-chip.
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## Recap on scientific applications' performance
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
We will start with a brief recap on the peak performance of current hardware and on performance evaluation of iterative stencil-based PDE solvers.
"""



using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

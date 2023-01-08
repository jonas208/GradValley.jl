using Documenter
using GradValley

makedocs(
    sitename = "GradValley.jl",
    format = Documenter.HTML(),
    modules = [GradValley]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#

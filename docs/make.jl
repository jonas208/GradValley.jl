using Documenter
using GradValley

makedocs(
    sitename = "GradValley.jl",
    format = Documenter.HTML(),
    modules = [GradValley],
    pages = ["Home" => "index.md", "Installation" => "installation.md", "Getting Started" => "getting_started.md", "Reference" => "reference.md", "Tutorials and Examples" => "tutorials_and_examples.md", "(Pre-Trained) Models" => "(pre-trained)_models.md", "Learning" => "learning.md"]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/jonas208/GradValley.jl.git"
    devurl=""
)

# =====================================================================
# MOADyna.jl documentation build driver
# =====================================================================
#
# Local build:
#   julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   julia --project=docs docs/make.jl
#
# `prettyurls` is enabled only under CI so that a local build writes
# `docs/build/<page>.html` files that open directly in a browser
# (offline, no web server). `deploydocs` targets GitHub Pages for the
# public repository and is a no-op when run outside CI.

# Headless GR for the plotted tutorials: `@example` blocks that call
# `using Plots; plot(...)` render through GR with no display attached.
# `GKSwstype = "100"` selects GR's off-screen (file) workstation so the
# build runs in CI / over SSH without an X server.
ENV["GKSwstype"] = "100"

using Documenter
using MOADyna

# All submodules are top-level under MOADyna (see src/MOADyna.jl); list every
# one so `@autodocs` Library pages and `checkdocs` see the full surface.
const MOADYNA_MODULES = [
    MOADyna,
    MOADyna.Algebra,
    MOADyna.Bases,
    MOADyna.ED,
    MOADyna.Responses,
    MOADyna.Spectroscopy,
    MOADyna.Shells,
    MOADyna.AtomicParameters,
    MOADyna.PointGroups,
    MOADyna.QuantyIO,
    MOADyna.Diagnostics,
    MOADyna.Units,
    MOADyna.Gradients,
]

const PAGES = [
    "Home" => "index.md",
    "Changelog" => "changelog.md",
    "Manual" => [
        "Introduction"                   => "man/intro.md",
        "Operator algebra"               => "man/algebra.md",
        "Hilbert spaces & bases"         => "man/bases.md",
        "Exact diagonalization"          => "man/ed.md",
        "Responses"                      => "man/responses.md",
        "Spectroscopy"                   => "man/spectroscopy.md",
        "Multiplets & standard operators" => "man/multiplets.md",
        "Point groups"                   => "man/pointgroups.md",
        "Atomic parameters"              => "man/atomic.md",
    ],
    # NOTE: a top-level "Tutorials" bucket is a deliberate adaptation —
    # TensorKit folds its tutorial into the Manual; MOADyna promotes worked
    # examples to their own section for the spectroscopy / Quanty audience.
    "Tutorials" => [
        "Getting started"      => "tut/getting_started.md",
        "NiO: XAS, RIXS, and nIXS" => "tut/nio.md",
    ],
    "Library" => [
        "Algebra"          => "lib/algebra.md",
        "Bases"            => "lib/bases.md",
        "ED"               => "lib/ed.md",
        "Responses"        => "lib/responses.md",
        "Spectroscopy"     => "lib/spectroscopy.md",
        "Shells"           => "lib/shells.md",
        "AtomicParameters" => "lib/atomic.md",
        "PointGroups"      => "lib/pointgroups.md",
        "QuantyIO"         => "lib/quantyio.md",
        "Diagnostics"      => "lib/diagnostics.md",
        "Units"            => "lib/units.md",
        "Gradients"        => "lib/gradients.md",
        "Index"            => "lib/index_page.md",
    ],
    "Appendix" => [
        "Conventions"        => "app/conventions.md",
        "Point-group theory primer" => "app/point_groups.md",
        "Coming from Quanty" => "app/from_quanty.md",
        "Acknowledgments"    => "app/acknowledgments.md",
    ],
]

makedocs(;
    sitename = "MOADyna.jl",
    authors  = "Yi Lu and contributors",
    modules  = MOADYNA_MODULES,
    # Pin the GitHub repo explicitly for source links (matches
    # `canonical` + `deploydocs`).
    repo     = Documenter.Remotes.GitHub("CorrelatedSpectra", "MOADyna.jl"),
    format   = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical  = "https://correlatedspectra.github.io/MOADyna.jl/stable",
        edit_link  = "main",
        # `assets/logo.svg` is auto-detected; an SVG favicon link is wired
        # in the Stage-3 polish step (Documenter's `assets` list only takes
        # .css/.js/.ico classes).
    ),
    pages    = PAGES,
    doctest  = true,
    checkdocs = :exports,
    # The build is STRICT: doctests, missing docstrings, and unresolved
    # cross-references are all errors. Every `@example` / `@repl` block
    # runs at build time.
)

# No-op locally; deploys from CI to GitHub Pages.
deploydocs(;
    repo      = "github.com/CorrelatedSpectra/MOADyna.jl",
    devbranch = "main",
    versions  = ["stable" => "v^", "dev" => "dev"],
)

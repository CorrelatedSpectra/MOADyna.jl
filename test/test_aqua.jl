# Package-quality checks (Aqua.jl).
#
# These cover structural properties the physics tests cannot: method
# ambiguities, unbound type parameters, undefined exports, stale or
# uncapped dependencies, type piracy, and persistent tasks.
#
# `test_all` runs with no exclusions — if a check starts failing, fix the
# cause rather than disabling the check.

using Aqua

@testset "Aqua quality assurance" begin
    Aqua.test_all(MOADyna)
end

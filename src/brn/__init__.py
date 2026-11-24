"""DC-SIIG: decentralized compressed implicit-gradient learning.

This package is self-contained. It depends on the upstream project
``Mixed_Autonomy_Traffic_Assignment`` READ-ONLY, for the equilibrium follower,
the reduced-KKT machinery and the centralized reference solver. Nothing in
this package ever writes into the upstream repository.
"""

from .paths import ensure_upstream_on_path, tntp_dir, upstream_root
from .partition import (
    RegionDecomposition,
    build_partition,
    spectral_partition,
    communication_graph,
    decompose,
    node_block_partition,
)
from .controls import reference_controls, uniform_av_reduced, uniform_logits
from .assembly import LocalSystem, assemble_local_systems, centralized_system
from .compression import (
    Compressor,
    ErrorFeedback,
    StochasticQuantizer,
    TopK,
    UniformQuantizer,
    build_compressor,
)
from .game import (
    CoalitionStructure,
    build_structure,
    grand_coalition,
    regional_road_objectives,
    regional_objectives,
    singletons,
)
from .outer import (
    DCSIIGOptions,
    DCSIIGResult,
    DCSIIGTrace,
    GameResult,
    run_dcsiig,
    run_regional_game,
)
from .distributed import (
    AdjointSolution,
    CommunicationLedger,
    DistributedOperator,
    distributed_cg,
    local_jacobi_preconditioner,
)

__all__ = [
    "AdjointSolution",
    "CoalitionStructure",
    "DCSIIGOptions",
    "GameResult",
    "build_structure",
    "grand_coalition",
    "regional_road_objectives",
    "regional_objectives",
    "run_regional_game",
    "singletons",
    "DCSIIGResult",
    "DCSIIGTrace",
    "run_dcsiig",
    "CommunicationLedger",
    "Compressor",
    "DistributedOperator",
    "ErrorFeedback",
    "LocalSystem",
    "RegionDecomposition",
    "StochasticQuantizer",
    "TopK",
    "UniformQuantizer",
    "assemble_local_systems",
    "build_compressor",
    "centralized_system",
    "communication_graph",
    "decompose",
    "distributed_cg",
    "ensure_upstream_on_path",
    "local_jacobi_preconditioner",
    "node_block_partition",
    "reference_controls",
    "spectral_partition",
    "build_partition",
    "uniform_av_reduced",
    "uniform_logits",
    "tntp_dir",
    "upstream_root",
]

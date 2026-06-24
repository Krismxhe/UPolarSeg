"""Research model layer.

User-owned research implementations (e.g. ModularUNet, and future PolarSeg /
BoundaryAwareUNet / CrossViewSeg) live here. They are built through
:func:`src.models.research.registry.build_research_model`, which is the only
entry point the model factory uses for ``provider=research``.

Research code must not import baseline internals.
"""

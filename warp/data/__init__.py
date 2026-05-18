"""sto-warp reference-data layer.

`warp.data.cargo` exposes the SETS-shaped item / ship / trait / BOFF
metadata that the recognition pipeline needs. Data is fetched on
demand from STOCD/SETS-Data, cached per-user under
`$XDG_CONFIG_HOME/warp/cache/`, with a frozen `warp/data/baseline/`
snapshot bundled in the wheel for offline first-run.
"""

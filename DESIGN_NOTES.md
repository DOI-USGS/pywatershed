<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Design notes](#design-notes)
  - [Process variants as subclasses](#process-variants-as-subclasses)
    - [Cases](#cases)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Design notes

A ledger of design problems observed while working on the code. Each
entry records a concrete case: the symptom, what it took to work
around, and the root cause in one line. The aim is to accumulate
enough cases to weigh design options against evidence, not to fix
anything here. Add entries as they arise.

## Process variants as subclasses

A physics option (depression storage on or off, cascades on or off,
agriculture on or off) is expressed as a subclass of a process, and
the parent's `__init__` does the wiring from declarations the child
must override in several places at once. Two consequences recur:

- **Declarations are spread and unchecked.** An input lives in the
  `__init__` signature, its docstring, `get_inputs()`, and the
  `super().__init__()` forward. Restart variables, init values, and
  parameters have the same shape. Nothing checks that they agree.
- **The parent decides things the child needs to veto or pre-empt.**
  The instance name, kernel parallelization, and how `None` inputs
  are filled are all set in the parent from the parent's point of view.

The cost also compounds: three binary options already give
`PRMSRunoff`, `PRMSRunoffNoDprst`, `PRMSRunoffCascadesNoDprst`, and
`PRMSRunoffAg`; each new option doubles the number of leaf classes.

### Cases

- **`stream_seg_in` accepted and silently dropped** (PR 407 review).
  `PRMSSoilzone` and `PRMSSoilzoneNoDprst` took it in `__init__` but
  neither listed it in `get_inputs()`, so `_set_inputs` ignored it.
  Only `PRMSSoilzoneCascadesNoDprst` uses it, forwarding through the
  parent's signature. Workaround: a guard in `Process._set_inputs`
  that raises on any argument naming a model variable that is not in
  `self.inputs`. The parent's docstring now says the argument is a
  pass-through slot for the child (2026-09-29): a second workaround,
  in prose, for the same declaration. Root cause: declarations spread
  and unchecked.
- **Parents guard `self.name` with `hasattr`** (PR 407 review). Cascade
  children set `self.name` before calling `super().__init__()`, and
  the parent overwrote it. Workaround: `if not hasattr(self, "name")`
  in `PRMSRunoff` and `PRMSSoilzone`. The guard only works before
  `super().__init__()`, because `Process.__init__` sets a default
  name; placed after it, both parents were named `Process` and
  collided on their budget output file (found 2026-09-04). Root
  cause: parent decides. Resolved 2026-10-01: `Process.__init__` names
  every instance for its class (`type(self).__name__`), the guards are
  gone, and no subclass assigns `name` any more. The last step was
  forced: a parent's assignment after `super().__init__()` overwrites
  the default for its children too (`PRMSGroundwaterNoDprst` was
  briefly named `PRMSGroundwater`, and `PRMSAtmosphereTranspFrost` had
  always been named `PRMSAtmosphere`).
- **`PRMSRunoffAg` skips its parent's `__init__`** (PR 407 review).
  `PRMSRunoff.__init__` calls `self._set_inputs(locals())` with its own
  locals, and `_set_inputs` loops over the child's `get_inputs()`; the
  three ag inputs (`ag_soil_moist_prev`, `ag_soil_rechr_prev`,
  `ag_frac`) are not in the parent's signature, so `args[...]` would
  raise `KeyError`. `PRMSRunoffAg.__init__` therefore calls
  `ConservativeProcess.__init__` directly (`prms_runoff_ag.py`, search
  "grandparent") and repeats the rest of the parent's `__init__` by
  hand. The copy drifts: when the active-HRU setup
  (`_set_active_hrus`, `_mask_inactive_hrus`,
  `_set_budget(active_mask=...)`) was added to the parent, the copy
  did not get it, and `PRMSRunoff.basin_init` grew a
  `hasattr(self, "_wh_active_hrus")` patch instead (both fixed
  2026-09-29). Root cause: declarations spread (`locals()` ties input
  setup to the frame whose signature declares the inputs) and parent
  decides (a child with extra inputs can only skip the parent
  wholesale).
- **`_nb_parallel_ok` class attribute** (PR 407 review). Cascade kernels
  must never run under numba `prange`, but the parent chooses the
  kernel's parallel flag. Workaround: a class attribute, True on
  parents and False on cascade children, and-ed into the decision.
  Root cause: parent decides.
- **`None` inputs filled with zeros for no-dprst children.**
  `prms_soilzone.py` (search "hacky dprst_flag == False approach")
  replaces every `None` input with a zero array when `_dprst_flag` is
  False, because the child passes `dprst_evap_hru=None` and
  `dprst_seep_hru=None` through a signature that still requires them.
  `prms_groundwater.py` has the same loop for `dprst_seep_hru`.
  Root cause: parent decides, and declarations spread.
- **Depression-storage restart variables on a no-dprst class**
  (PR 407, resolved there). `PRMSRunoffCascadesNoDprst` inherited
  `get_restart_variables` from `PRMSRunoff`, so it listed `dprst_*`
  variables the class does not carry, and the cascade classes had no
  `restart_read` or `restart_write` at all. Resolved by giving each
  no-dprst and cascade child its own restart list (two impervious
  storages for runoff, four reservoir storages for soilzone) and the
  cascade children the restart arguments, forwarded to the parent;
  `test_restart_processes.py` now covers them. The fix is one more
  spread declaration: four leaf classes carry the same two lists by
  hand. Root cause: declarations spread.
- **Copy-paste `_calculate`** (PR 407 review, Quality). The cascade
  children repeat their NoDprst siblings' ~90-keyword kernel call
  almost verbatim (`prms_runoff_cascades_no_dprst.py` vs
  `prms_runoff_no_dprst.py`; same for soilzone). Every change to a
  kernel signature is hand-mirrored in three places; PR 407 made one.
  Not worked around. Root cause: parent decides (the parent owns the
  kernel call, so a child can only replace it, not extend it).
- **Doubled startup work** (PR 407 review, Quality). The cascade
  children rerun `_set_inputs`, `_set_options`, `_set_budget` and
  `basin_init` after `super().__init__()` already ran them
  (`prms_runoff_cascades_no_dprst.py`, search `basin_init`; soilzone
  reruns `_set_budget`), because the child must preprocess parameters
  before the parent wires them and the parent's `__init__` cannot be
  entered halfway. When the parameters are not already preprocessed,
  `preprocess_cascade_params` also runs once per class, so twice per
  model (three times with the groundwater cascade class). Not worked
  around. Root cause: parent decides.
- **Two off-switch conventions for one option** (PR 407 review,
  Quality). `PRMSRunoff` turns cascades off with per-step NaN sentinel
  arrays (`prms_runoff.py`, search `nan_array`) plus a compiled dummy
  kernel, and re-derives the flag each step from
  `np.isnan(ncascade_hru).all()`; `PRMSSoilzone` passes `None`. The
  parent's kernel carries arguments only the child fills. Not worked
  around. Root cause: declarations spread (the option lives in the
  kernel signature of a class that does not have the option).
- **A child's input threaded through the parent's signature**
  (feat_gw_cascades, 2026-09-09). `PRMSGroundwater.__init__` gained
  `stream_seg_in: adaptable = None` that the parent never reads,
  because `Process._set_inputs(locals())` runs inside the parent's
  `__init__`, so any input of a child must be an argument of the
  parent. `PRMSSoilzone` already had the same shape for the same
  variable (first case above); the B13 guard is what keeps the
  parent from silently dropping it. Root cause: declarations spread.
- **Third convention for `self.name`** (feat_gw_cascades,
  2026-09-09). `PRMSGroundwater` has no `hasattr` guard, so
  `PRMSGroundwaterCascadesNoDprst` sets `self.name` *after*
  `super().__init__()` and reruns `_set_budget` so the budget carries
  the child's name; the runoff and soilzone children set it *before*
  and rely on the guard. Same problem, now solved two ways. Root
  cause: parent decides.

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

- **`stream_seg_in` accepted and silently dropped** (PR 407, B13).
  `PRMSSoilzone` and `PRMSSoilzoneNoDprst` took it in `__init__` but
  neither listed it in `get_inputs()`, so `_set_inputs` ignored it.
  Only `PRMSSoilzoneCascadesNoDprst` uses it, forwarding through the
  parent's signature. Workaround: a guard in `Process._set_inputs`
  that raises on any argument naming a model variable that is not in
  `self.inputs`. Root cause: declarations spread and unchecked.
- **Parents guard `self.name` with `hasattr`** (PR 407, B6). Cascade
  children set `self.name` before calling `super().__init__()`, and
  the parent overwrote it. Workaround: `if not hasattr(self, "name")`
  in `PRMSRunoff` and `PRMSSoilzone`. The guard only works before
  `super().__init__()`, because `Process.__init__` sets a default
  name; placed after it, both parents were named `Process` and
  collided on their budget output file (found 2026-09-04). Root
  cause: parent decides.
- **`_nb_parallel_ok` class attribute** (PR 407, B2). Cascade kernels
  must never run under numba `prange`, but the parent chooses the
  kernel's parallel flag. Workaround: a class attribute, True on
  parents and False on cascade children, and-ed into the decision.
  Root cause: parent decides.
- **`None` inputs filled with zeros for no-dprst children.**
  `prms_soilzone.py` (search "hacky dprst_flag == False approach")
  replaces every `None` input with a zero array when `_dprst_flag` is
  False, because the child passes `dprst_evap_hru=None` and
  `dprst_seep_hru=None` through a signature that still requires them.
  Root cause: parent decides, and declarations spread.
- **Depression-storage restart variables on a no-dprst class.**
  `PRMSRunoffCascadesNoDprst.get_restart_variables` lists `dprst_*`
  variables the class does not carry. Restart is not expected for
  the cascade processes at all; their signatures have no
  `restart_read` or `restart_write`. Root cause: declarations spread.

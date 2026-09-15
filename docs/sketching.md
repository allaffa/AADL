# Anderson sketching

For a parameter group with `n` coordinates and an Anderson history with `m`
difference columns, the unsketched least-squares matrix has shape `n x m`.
AADL can estimate the mixing coefficients from selected rows while still
assembling the final candidate from the complete parameter history. This
reduces the tall least-squares work without returning a partial model update.

Sketching is opt-in. The defaults use `sketch_fraction=1.0`, which is the
original full Anderson calculation.

## Policies

| Policy | Growth signal | Requires a closure | Persistent adaptation |
| --- | --- | --- | --- |
| `fixed` | None | No | No |
| `adaptive` | Candidate rejected by the loss safeguard | Yes | No |
| `backward_error` | Omitted update energy or excessive LS condition; then loss rejection | No for algebraic checks; yes for loss checks | Yes |

`fixed` performs one attempt at `sketch_fraction`. With no closure, that
candidate is accepted because there is no objective value against which to
compare it.

`adaptive` starts at `sketch_fraction`. After a safeguarded rejection it
multiplies the fraction by `sketch_growth_factor`, stopping at
`sketch_max_fraction`. If every attempt is rejected, the plain optimizer step
is retained.

`backward_error` adds inexpensive algebraic checks before objective
acceptance. It remembers the fraction that last passed and starts the next
cycle there. After `sketch_successes_before_shrink` consecutive first-attempt
successes, it probes a fraction smaller by one growth step. This hysteresis
prevents the selected fraction from oscillating every cycle.

Set `sketch_max_retries` to bound the additional work. A value of `0` permits
only the initial attempt; `2` permits the initial attempt plus two larger
sketches. The default `None` permits growth through `sketch_max_fraction`.

## Coordinate strategies

| Strategy | Nested during growth | Selection cost | Intended use |
| --- | --- | --- | --- |
| `stratified` | No | O(s) | Low-memory general default |
| `random_nested` | Yes | O(n) permutation, cached for the cycle | Controlled random-growth experiments |
| `magnitude` | Yes | Top-k selection | Updates with concentrated residual energy |
| `block` | Yes | O(s) | Contiguous-access experiments and accelerators |

Nested means that a larger retry retains all coordinates from its preceding
attempt. `random_nested` caches one permutation per parameter group and
acceleration cycle instead of regenerating it at every retry. Its O(n) index
memory can outweigh the benefit for extremely large models; use `stratified`
when that matters. Magnitude sampling is deterministic for a given update but
can systematically focus on a subset of parameters. Block sampling minimizes
index fragmentation but can miss structure outside the selected interval.

## Backward-error surrogate

Let `b` be the latest optimizer update and `S` the selected coordinate rows.
AADL computes the relative omitted energy

```text
epsilon = ||(I - S^T S)b||_2 / ||b||_2
        = sqrt(1 - ||Sb||_2^2 / ||b||_2^2).
```

The sketch passes the energy check when

```text
epsilon <= sketch_energy_tolerance / max(1, L),
```

where the running local sensitivity estimate is

```text
L = max ||delta update||_2 / ||previous update||_2.
```

AADL then checks that the estimated condition number of the sketched
least-squares matrix does not exceed `sketch_condition_limit`. The QR kernel
uses its already-computed triangular factor. The normal-equation kernel uses
the already-computed Gram matrix and converts its condition to the equivalent
least-squares condition. No additional tall factorization is performed.

When `sketch_rescale=True`, selected rows of both sides of the least-squares
problem are multiplied by `sqrt(n / s)`. This leaves an unregularized solution
unchanged while preventing Tikhonov regularization from automatically becoming
stronger merely because fewer random rows were retained. For magnitude-based
sampling this is a pragmatic normalization, not an unbiased importance weight.

These quantities are practical proxies for backward stability, not a proof of
accuracy for a nonlinear stochastic optimizer. When a closure is supplied,
the ordinary loss safeguard remains the final authority and restores the plain
optimizer step after rejection.

## Configuration

```python
AADL.accelerate(
    optimizer,
    acceleration_type="anderson",
    sketch_fraction=0.1,
    sketch_policy="backward_error",
    sketch_growth_factor=2.0,
    sketch_max_fraction=1.0,
    sketch_energy_tolerance=0.1,
    sketch_condition_limit=1e8,
    sketch_successes_before_shrink=3,
    sketch_strategy="random_nested",
    sketch_max_retries=3,
    sketch_lipschitz_mode="ema",
    sketch_lipschitz_decay=0.9,
    sketch_rescale=True,
    sketch_seed=0,
)
```

With these values, attempted fractions can progress as
`0.1 -> 0.2 -> 0.4 -> 0.8 -> 1.0`. The effective coordinate count is also
bounded below by the number of least-squares columns so that the QR problem is
not underdetermined.

`sketch_energy_tolerance` controls a norm, not a fraction of coordinates. A
value of `0.1` requires at least 99% of the squared update energy to be
retained. Diffuse updates may therefore require a much larger coordinate
fraction than sparse or strongly concentrated updates.

Use `optimizer.acc_last_sketch_fraction` for lightweight observation of the
last attempted fraction. The controller's remembered fraction, success streak,
and Lipschitz estimate are reset whenever `reset_acceleration_history()` is
called, including after native periodic model averaging.

`optimizer.acc_sketch_last_diagnostics` contains the most recent controller
trace. Each attempt records its fraction, result (`energy_rejected`,
`condition_rejected`, `loss_rejected`, or `accepted`), and per-group row count,
energy estimate, energy limit, and condition estimate. This is intended for
profiling and experiment logging; training logic should not depend on it.

## Distributed training

Sampling and algebraic indicators are local because histories and Anderson
coefficients are local. With `safeguard=False`, `backward_error` can still
increase the local sketch using its energy and conditioning checks. At a native
model-averaging boundary, `average_and_accept()` evaluates the globally
averaged candidate using the configured vote or mean-loss policy.

A global rejection does not replay local Anderson calculations with larger
sketches. It retains the globally averaged plain branch and resets Anderson and
controller history. Every rank must continue to enter distributed collectives
in the same order.

## Tuning guidance

- Start with `sketch_fraction` between `0.05` and `0.2` for large parameter
  groups, then profile on the target hardware.
- Keep `sketch_max_fraction=1.0` when a full-Anderson fallback is affordable.
- Increase `sketch_energy_tolerance` if updates are diffuse and the controller
  almost always reaches the full system.
- Lower `sketch_condition_limit` to be more conservative with unstable mixing
  coefficients; regularization and history filtering remain complementary.
- Increase `frequency` when forward safeguard evaluations, rather than the
  least-squares calculation, dominate runtime.
- Use `sketch_lipschitz_mode="running_max"` for a conservative bound. Use
  `"ema"` when an early transient otherwise forces oversized sketches for the
  remainder of training; `sketch_lipschitz_decay` controls its memory.

Random indexing is not free, particularly on accelerators. AADL uses ordered
stratified samples and avoids constructing a full random permutation, but the
best fraction is still model-, backend-, and hardware-dependent.

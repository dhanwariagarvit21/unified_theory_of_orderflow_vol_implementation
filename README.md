# Bivariate Hawkes model of BTCEUR trade flow (Binance spot, October 2026)

A GPU-accelerated bivariate Hawkes process fitted to one month of
aggressor-signed trades, with an out-of-sample test and time-rescaling
(QQ) diagnostics. The work is motivated by the "unified theory of order
flow" line of research (core flow, reaction flow, Hawkes processes), but
**this repository tests only a single-exponential bivariate model on one
instrument and one month**. Section 9 says what that does and does not
tell us about the paper.

## Contents

-   [Summary of findings](#1-summary-of-findings)
-   [Data](#2-data)
-   [Model](#3-model)
-   [Implementation and validation](#4-implementation-and-validation)
-   [Results: full-month fit](#5-results-full-month-fit)
-   [Out-of-sample evaluation](#6-out-of-sample-evaluation)
-   [Time-rescaling diagnostics and QQ
    plots](#7-time-rescaling-diagnostics-and-qq-plots)
-   [Interpretation](#8-interpretation)
-   [Limitations](#9-limitations-and-what-this-does-not-show)
-   [Next steps](#10-next-steps-not-yet-done)
-   [Reproducing](#11-reproducing)
-   [Appendix: formulas](#appendix-formulas)

------------------------------------------------------------------------

## 1. Summary of findings

1.  **Same-side excitation dominates cross-side excitation by about two
    orders of magnitude.** Buy→buy and sell→sell branching are 0.49 and
    0.43; buy→sell and sell→buy are 0.004 to 0.005 (ratio about 104 to 1
    on the full month, 200 to 1 and 420 to 1 when fitted on the first 21
    days only).
2.  **All the clustering is at the millisecond scale.** The fitted decay
    rate is β ≈ 413 per second (time constant ≈ 2.4 ms, half-life ≈ 1.7
    ms).
3.  **The model is stable and far from critical at this scale:**
    spectral radius 0.49 (0.44 on the 21-day fit).
4.  **Out of sample the Hawkes model beats a Poisson model by about 3.4
    nats per event**, and beats a Poisson model that is *given the
    test-period rates* by about 3.3 nats per event, so the gain comes
    from clustering and not from the average rate.
5.  **The time-rescaling QQ plots reject the model as a complete
    description.** The rescaled intervals have mean 0.83 (buy) and 0.80
    (sell), not 1. The exponential QQ plots bend sharply above the
    diagonal in the upper tail, and the uniform QQ plots show a plateau
    whose height equals 1 − exp(−branching ratio). This points to (a) a
    single memory time scale being too restrictive and (b) a constant
    baseline being too restrictive.
6.  **The cross-excitation values are not stable across fitting
    windows** (see Section 5), so the claim is "cross excitation is much
    smaller than same-side excitation", not "cross excitation equals
    0.0047".
7.  An earlier version of the likelihood had a bug that produced a
    spurious one-directional 0.46 buy→sell excitation. It was replaced
    and validated (Section 4). Numbers from that version are discarded.

------------------------------------------------------------------------

## 2. Data

  ------------------------------------------------------------------------
  Item        Value
  ----------- ------------------------------------------------------------
  Source file `BTCEUR-trades-2026-10.csv` (Binance spot trade archive)

  Columns     trade_id, price, qty, quote_qty, timestamp, is_buyer_maker,
              is_best_match

  Period      2026-10-01 00:00:01 UTC to 2026-10-31 23:59:58 UTC (T =
              2,678,397 s ≈ 31.0 days)

  Timestamp   microseconds
  unit        

  Raw trades  685,061 (no missing values after cleaning)
  ------------------------------------------------------------------------

**Signing.** `is_buyer_maker == True` means the buyer was the resting
order, so the **aggressor was a seller** (direction −1). `False` means
an aggressive buyer (+1).

**Event construction.** Prints are merged on **(timestamp, side)**: one
event per unique timestamp per side, with price ignored (the model uses
only time and side).

  -----------------------------------------------------------------------
                      Count
  ------------------- ---------------------------------------------------
  Buy events          255,423

  Sell events         280,726

  Total events        536,149 (78.3% of raw prints; 149k prints shared an
                      exact timestamp and side with another print)

  Unique event        536,074
  timestamps          

  Timestamps with     75
  both a buy and a    
  sell                

  Mean rate           0.0954 buys/s and 0.1048 sells/s
  -----------------------------------------------------------------------

**Note on merging.** Merging on exact microsecond equality removes
prints of one order that were stamped at the same microsecond. Prints of
the *same* taker order stamped a few microseconds or milliseconds apart
are **not** merged and may still appear as same-side clustering (see
Section 9).

------------------------------------------------------------------------

## 3. Model

Two event types, `b` (buy) and `s` (sell), with intensities:

``` text
λ_b(t) = μ_b + α_bb · E_b(t) + α_bs · E_s(t)
λ_s(t) = μ_s + α_sb · E_b(t) + α_ss · E_s(t)

E_k(t) = Σ_{t_i < t, type k}  exp(−β (t − t_i))        (one shared decay rate β)
```

-   `α_ij` is the instantaneous jump in the intensity of the first type
    caused by one event of the second type (so `α_bs` is the effect of a
    **sell** on the **buy** intensity). Units: events per second.
-   **Branching matrix** `B[i][j] = α_ij / β` = expected number of
    type-i events triggered by one type-j event. Reported as
    `[[bb, bs], [sb, ss]]` with rows = target intensity, columns =
    source event.
-   **Stability.** The optimiser works on `z` and maps to parameters so
    that each row sum of `B` lies in (0, 1): `row_sum = sigmoid(z)`,
    `split = sigmoid(z)`, `β = exp(z)`. The spectral radius of a
    non-negative matrix is at most its largest row sum, so every
    candidate is stable. The bounds allow row sums up to about 0.9999,
    so near-criticality was available to the optimiser and was not
    chosen.
-   Estimation by maximum likelihood (L-BFGS-B, 6 to 12 starting points,
    analytic gradients from PyTorch autograd).

------------------------------------------------------------------------

## 4. Implementation and validation

**Computing the excitation in O(N).** The excitation is a sum over past
events, `E_n = Σ_{k<n} c_k exp(−β (t_n − t_k))`. Writing
`exp(−β(t_n − t_k)) = exp(−β t_n) · exp(β t_k)` turns it into a running
sum, evaluated in the log domain with `torch.logcumsumexp`:

``` text
log E_n = −β t_n + logcumsumexp_k( β t_k + log c_k )      (over earlier events of that type only)
```

No N×N matrix is built. The data are processed in chunks with local
times (so `β t` stays small) and the excitation state is carried between
chunks, including the compensator across the gaps.

**Checks that were run (all passed):**

  -----------------------------------------------------------------------------------------------------
  Check                                                        Result
  ------------------------------------------------------------ ----------------------------------------
  Excitation vs brute-force sum                                match to \~1e-8

  Gradient finite and non-NaN                                  passed

  `NLL(whole) == NLL(train) + NLL(test given carried state)`   difference −5.6e-11
  on a 5,000-event slice                                       

  Same fit with `batch_size` 5,000, 10,000, 15,000             NLL 105,303.243 to 105,303.247; β
                                                               412.675, 412.686, 412.680; branching
                                                               entries identical to 3 to 4 digits

  Several random starts                                        NLL agrees to \~0.01 (relative 1e-7); no
                                                               sign of multiple optima for this model
                                                               class

  Recomputed training NLL equals the fit's own NLL             `train_nll_matches_fit = True`

  Fitted stationary rates reproduce the data                   0.095361 and 0.104811 vs observed
                                                               0.095364 and 0.104811 events/s
  -----------------------------------------------------------------------------------------------------

**Limits of these checks.** The excitation check used β = 50, far below
the fitted β ≈ 413, so at most test points the excitation was ≈ 0 and
the comparison is only informative at the first few events. The
train/test carried state at the cutoff is ≈ 0 with a 2 ms memory
(2.8e-142 for buys), so the out-of-sample run does not exercise the
carry logic; the slice identity check does.

**Bug history.** A first version computed the excitation with a scan
that passed the buy and sell streams as the two components of a single
recurrence, so the true decay factor was never applied. Its fit showed
one column of the branching matrix near zero and a 0.46 buy→sell entry.
On simulated, perfectly symmetric data the buggy code gave
`[[0.25, 0], [0.35, 0]]` against a true `[[0.30, 0.15], [0.15, 0.30]]`.
The corrected code recovers the symmetric truth. All results in this
README use the corrected code.

------------------------------------------------------------------------

## 5. Results: full-month fit

  Parameter                 Value
  ------------------------- ---------------------------
  μ_buy, μ_sell             0.0480, 0.0591 per second
  α_bb, α_bs                202.64, 1.954 per second
  α_sb, α_ss                1.705, 178.40 per second
  β                         412.68 per second
  Negative log-likelihood   105,303.24

**Branching matrix** `[[bb, bs], [sb, ss]]`:

``` text
[[0.4910, 0.0047],
 [0.0041, 0.4323]]          spectral radius = 0.4914
```

  -----------------------------------------------------------------------
  Derived quantity                  Buy                Sell
  --------------------------------- ------------------ ------------------
  Same-side branching               0.491              0.432

  Cross-side branching              0.0047 (sell→buy)  0.0041 (buy→sell)

  Same : cross ratio                104 : 1            105 : 1

  Mean cluster size 1/(1 − B_ii)    1.96               1.76

  Memory time constant 1/β          2.4 ms             2.4 ms

  Half-life ln2/β                   1.7 ms             1.7 ms
  -----------------------------------------------------------------------

**Same-side share.** Normalising each column by its sum, same-side
excitation is about 98 to 99% of the total (0.4910 / (0.4910 + 0.0041) ≈
0.992). In the paper's notation this corresponds to a very high κ (≈
0.98), but note that the paper defines κ under the assumption that the
*sum* channel is critical; here the sum is ≈ 0.49, so only the *ratio*
is comparable.

### Stability of the estimates across windows

  ----------------------------------------------------------------------------------
  Fit                    β       bb      ss      bs       sb       Spectral radius
  ---------------------- ------- ------- ------- -------- -------- -----------------
  Full month (31 days)   412.7   0.491   0.432   0.0047   0.0041   0.491

  First 21 days          479.5   0.444   0.413   0.0020   0.0011   0.444
  ----------------------------------------------------------------------------------

The same-side entries move by about 10% and β by about 15%. The cross
entries move by a factor of 2 to 4, so only their order of magnitude can
be stated. **No standard errors have been computed yet.**

------------------------------------------------------------------------

## 6. Out-of-sample evaluation

**Design.** Train on the first 21 days (1,814,400 s), test on the
remaining 10 days (864,000 s). Parameters are fitted on the training
window only. The training excitation state at the cutoff is carried into
the test window.

  -----------------------------------------------------------------------
  Metric                            Value
  --------------------------------- -------------------------------------
  Training events                   314,031 (0.173 events/s)

  Test events                       222,118 (0.257 events/s, **49%
                                    busier**)

  Train NLL (per event)             170,520.4 (0.543)

  Test NLL (per event)              −61,399.4 (−0.276)

  Poisson NLL with training rates   692,728.2

  Poisson NLL with test rates       677,309.0
  (oracle)                          

  **Gain over Poisson, per event**  **3.395 nats**

  Gain over oracle Poisson, per     3.326 nats
  event                             
  -----------------------------------------------------------------------

**Reading these numbers.**

-   **NLL per event is not comparable between periods.** The densities
    are in events per second, so the log-likelihood can be positive (the
    NLL negative) and shifts with the event rate. The test period is 49%
    busier, so its per-event NLL is lower for that reason alone. The
    *difference from a Poisson baseline* is the meaningful number. The
    consistency check holds: train + test NLL under the 21-day
    parameters is 109,121, against 105,303 for the in-sample full-month
    fit.
-   **The gain of 3.4 nats per event is large because the clustering is
    large.** A gain of 3.3 over the oracle Poisson shows that it comes
    from the ms-scale clustering and not from the test period's higher
    rate.
-   **The Poisson baseline is deliberately weak.** A Poisson process
    with a time-varying rate (for example hourly) is the fairer
    comparison and has not been run.

------------------------------------------------------------------------

## 7. Time-rescaling diagnostics and QQ plots

**Method.** If the model is correct, the integrated intensity between
consecutive events of the same type, `z_i = ∫_{t_{i−1}}^{t_i} λ(s) ds`,
is i.i.d. Exp(1). Then `u_i = 1 − exp(−z_i)` is Uniform(0, 1). The
intervals were computed on the test window with the 21-day parameters
(103,716 buy intervals, 118,402 sell intervals). The interval code was
reviewed and is correct: it uses the state after each event, the
closed-form integral of the exponential decay, and cumulative
differences at the events of one type.

  ----------------------------------------------------------------------------
                Buy     Sell    Expected if the model were right
  ------------- ------- ------- ----------------------------------------------
  Mean of z     0.832   0.796   1.000

  Maximum of z  ≈ 35    ≈ 47    ≈ 12.2 and 12.4 (largest of n Exp(1) draws)
  ----------------------------------------------------------------------------

Mean z below 1 means the observed event count exceeds the model's
predicted count by about 20% (buys) and 26% (sells) in the test window.

### Exponential QQ plots

**Buy exponential QQ plot** · **Sell exponential QQ plot**

  ------------------------------------------------------------------------
  Region        What is seen (values read     Meaning
                from the plots, approximate)  
  ------------- ----------------------------- ----------------------------
  Theoretical   Points sit slightly **below** More short rescaled
  quantiles up  the diagonal                  intervals than Exp(1),
  to ≈ 3                                      consistent with mean z \< 1

  Quantiles ≈ 4 Points curve **above** the    Heavier upper tail than
  to 8          diagonal                      exponential

  Quantiles ≈ 9 Empirical ≈ 20 to 35 (buy), ≈ A handful of very long
  to 12         20 to 47 (sell) at            intervals relative to what
                theoretical ≈ 9 to 12         the model expects
  ------------------------------------------------------------------------

### Uniform QQ plots

**Buy uniform QQ plot** · **Sell uniform QQ plot**

  ---------------------------------------------------------------------------------
  Region        What is seen                                Meaning
  ------------- ------------------------------------------- -----------------------
  Whole range   Curve lies **below** the diagonal, by up to The rescaled intervals
                about 0.15 to 0.2 in the middle             are stochastically too
                                                            small (mean z \< 1)

  **Plateau**   The curve is flat near **0.36 (buy)** and   A point-mass-like lump
                **0.34 (sell)** over a stretch of roughly   in z at one value
                0.35 to 0.5 theoretical probability         (explained below)
  ---------------------------------------------------------------------------------

### Why the plateau sits where it does

After an event of type k, the excitation adds a fixed amount to the
integrated intensity, `∫ α e^{−βs} ds = α/β`, which equals the branching
ratio. For any next same-side event that arrives *long after the
excitation has died* (gap ≫ 1/β ≈ 2 ms), the rescaled interval is
approximately:

``` text
z ≈ (branching ratio) + μ · gap
```

so every gap between a few milliseconds and about a second lands in a
narrow band just above the branching ratio, however different the gaps
are in real time. With the 21-day parameters the branching ratios are
0.444 (buy) and 0.413 (sell), which map to:

``` text
u = 1 − exp(−0.444) = 0.359   (buy)         u = 1 − exp(−0.413) = 0.338   (sell)
```

These match the plateau heights in the plots. This is a structural
signature of the model's *shape*: it spends the whole excitation within
about 2 ms, while the real inter-arrival gaps are spread across many
time scales. (This reading is derived from the fitted numbers and the
plots; the fraction of intervals in the band has not been counted
directly.)

### What the diagnostics say about the model

  ------------------------------------------------------------------------------
  Observation      Most likely cause                Status
  ---------------- -------------------------------- ----------------------------
  Plateau at 1 −   **One decay rate** (single       Strongly suggested by the
  exp(−branching   exponential, shared β) cannot    arithmetic above
  ratio)           represent memory at several time 
                   scales                           

  Mean z ≈ 0.8,    **Constant baseline μ** cannot   Consistent, not yet tested
  test busier by   follow changes in overall        
  49%              activity between periods         

  Very large z (35 Long quiet gaps where the model  Not yet tested. If
  and 47)          still expects events: a baseline baseline-only, z = 35 and 47
                   that is too high in quiet hours, correspond to gaps of about
                   or **gaps in the data**          756 s (buy) and 895 s (sell)
  ------------------------------------------------------------------------------

None of these are fixed by adding more data to the same model.

------------------------------------------------------------------------

## 8. Interpretation

-   **Hypotheses for the same-side clustering at about 2 ms** (not
    distinguished by this analysis):
    1.  one aggressive order consuming several resting orders, printed
        as several fills whose timestamps differ by more than the
        exact-timestamp merging removes;
    2.  algorithmic child orders sent in quick bursts in one direction;
    3.  latency-driven bursts across venues or order books.
-   **Why cross excitation is tiny.** A buy does not noticeably trigger
    a sell within milliseconds. At this time scale buys and sells behave
    almost like two separate processes.
-   **What a near-zero cross term does not show.** It says nothing about
    *longer-horizon* interaction. A single 2 ms exponential cannot see
    minute-scale or hour-scale dependence.

------------------------------------------------------------------------

## 9. Limitations and what this does not show

1.  **One instrument, one month, spot only.** BTCEUR is a thinner pair
    than the dollar-denominated ones and may be shaped by arbitrage with
    other books.
2.  **Trade prints, not orders.** The model counts executions. A single
    parent order can generate many prints.
3.  **Single exponential kernel.** The paper's setting is near-critical
    with long (power-law) memory. The spectral radius here, 0.49, is
    measured at the ms scale and **says nothing about near-criticality
    at longer scales**; the QQ plots indicate the model is misspecified
    in exactly the dimension the paper cares about.
4.  **Constant baseline.** No intraday or day-of-week pattern is
    modelled.
5.  **No uncertainty estimates.** In particular the cross terms (0.001
    to 0.005 across windows) need confidence intervals.
6.  **Merging rule.** Only exactly equal timestamps are merged.
    Sensitivity to a merge window has not been tested.
7.  **The out-of-sample baseline is weak** (homogeneous Poisson).
8.  **Not a trading result.** Nothing here shows a profitable strategy,
    and the model uses no price information.

------------------------------------------------------------------------

## 10. Next steps (not yet done)

1.  **Diagnostics on the existing fit**
    -   Mean z by hour of day and by day (should be ≈ 1 everywhere if
        the baseline were adequate).
    -   Locate the largest z values in time and check for data gaps or
        maintenance windows.
    -   Count the share of intervals in the plateau band (z within 0.9
        to 1.2 times the branching ratio).
    -   Lag-1 autocorrelation of z, and a KS statistic for Exp(1).
2.  **Model changes**
    -   Replace the single exponential by a **sum of 3 to 5
        exponentials** with rates spread from about 1 per millisecond to
        about 1 per hour (keeps the O(N) recursion), or a power-law
        kernel.
    -   Add a **time-varying baseline** (piecewise-constant by hour of
        day).
3.  **Robustness**
    -   Merge same-side prints within 100 µs, 1 ms and 5 ms and refit.
        If same-side excitation falls sharply, the clustering is largely
        order splitting and not herding.
    -   Day-level fits and a block bootstrap by day for the branching
        ratios, especially the cross terms.
    -   Check trade-ID contiguity and price levels of clustered prints.
4.  **Better baselines**: Poisson with an hourly rate; the univariate
    Hawkes model.
5.  **Link back to the paper**: estimate the sign-autocorrelation decay
    and the Hurst exponent of aggregate signed flow on the same month,
    and compare with what a kernel with long memory would imply.

------------------------------------------------------------------------

## 11. Reproducing

1.  Place `BTCEUR-trades-2026-10.csv` in `/content/` (or edit the path
    in the first data cell).
2.  Run the notebook cells in order: imports, data loading and signing,
    event construction, GPU tensors, the excitation and likelihood
    functions, the batched fit, the validation checks, the out-of-sample
    function, the time-rescaling function, then the QQ plots.
3.  Requirements: Python with PyTorch (CUDA GPU used here), NumPy,
    pandas, SciPy, matplotlib, seaborn.
4.  Do **not** pass the full-month fit to the out-of-sample function as
    `fit_result`; it would leak the test period.
5.  Restart the runtime after redefining the likelihood function to
    clear any stale definition.

------------------------------------------------------------------------

## Appendix: formulas

**Log-likelihood** over `[0, T]` with events `t_n` of type `k_n`:

``` text
log L = Σ_n log λ_{k_n}(t_n)  −  ∫_0^T (λ_b + λ_s) dt

∫_0^T (λ_b + λ_s) dt = (μ_b + μ_s) T
                        + (α_bb + α_sb)/β · Σ_{buy events}  (1 − e^{−β(T − t_i)})
                        + (α_bs + α_ss)/β · Σ_{sell events} (1 − e^{−β(T − t_i)})
```

**Carrying state between windows.** The state just after the last event
is decayed by `exp(−β · gap)` to the next window's first event, and the
compensator over the gap is
`(μ_b + μ_s) · gap + [(α_bb + α_sb) · E_b + (α_bs + α_ss) · E_s] · (1 − e^{−β · gap})/β`.

**Time-rescaled intervals.** Between consecutive buy events at `t_{i−1}`
and `t_i`: `z_i = ∫ λ_b ds`, evaluated in closed form from the
post-event excitation states and the decay factor `(1 − e^{−β Δt})/β`.

# Claim ledger

This ledger separates completed evidence from tempting but unsupported claims. It is the authority for revising the paper draft.

| Claim | Evidence | Status | Allowed wording |
|---|---|---|---|
| Static local score ambiguity is associated with online tracking error | ASAP healthy subset: combined Spearman `rho=+0.2063`, 27/40 healthy works; week4 recordings have the same sign | Supported, modest association | “weak/modest positive association”; not individual-event prediction |
| Two seconds of future context improves a past-position estimate | Same Bayesian model: +5.72 percentage points chroma, +4.68 points pitch | Supported in matched retrospective fixed-lag setting | “future context improved retrospective localization” |
| High ambiguity identifies when waiting helps | Ambiguity–context-gain relation was not positive; downstream policies did not beat strong baselines | Rejected | State explicitly that ambiguity did not select the benefit |
| semi-P3 supplies a general online warning | Exact identities require ground truth; natural precursor and cross-corpus transfer gates failed | Rejected | Use only as a conditional post-hoc mechanism analysis |
| Ambiguity-aware maps improve model selection | Independent utility test: maxsim did not change any of 25 work-level choices and both learned rules lost to global pitch selection | Rejected | Map is an audit/stress axis, not a selector |
| A two-second post-hoc validator reduces relocation decision regret on unseen symbolic works | Frozen ASAP lockbox: 61 events, 7 works; regret `.7003 -> .3344`, improvement CI `[.0331,.8235]`; all gates pass | Supported within the ASAP MIDI pipeline | “confirmed on work/performer-disjoint symbolic lockbox” |
| The symbolic result is statistically definitive | Only four nonzero work improvements; exact one-sided sign-flip `p=.0625` | Not supported | Report bootstrap and exact diagnostic together |
| The validator transfers to real audio | Vienna pilot: 5 events, no holds; prospective: 9 events and negative improvement | Rejected | “failed to transfer” |
| A simpler parameter-free audio evidence rule rescues transfer | Prospective 32 performances: precision `.60`, improvement `-.2349`, CI `[-.4698,0]` | Rejected | “prospective rescue failed” |
| Audio failure is only a bad classifier | All 14 events over 40 performances came from `outerhmm` and two of four works; zero `arzt` events | Rejected | Failure includes trigger coverage and candidate evidence |
| Delayed validation / multi-hypothesis recovery is novel | Arzt & Widmer 2010; Fox & Quinn 2007; Yamamoto et al. 2013; Nakamura et al. 2016 | Rejected | Novelty is limited to evaluation framing and frozen transfer audit |
| The wrapper is ready for deployment | No microphone/device latency or user study; audio failed | Rejected | Symbolic simulation result only |

## Formal ambiguity relation

The paper distinguishes rather than conflates two quantities:

\[
A_{score}(j)=\max_{k\in\mathcal R(j)}\kappa(S_j,S_k)
\]

is score-only structural ambiguity, while

\[
A_{dyn}(e,t)=H(\tilde p_t(o),\tilde p_t(d))/\log2
\]

is the unresolved posterior uncertainty of the origin/destination pair created by a realized relocation. Their bridge is

\[
\log\frac{p_t(d)}{p_t(o)}=\log\frac{q_t(d)}{q_t(o)}-\beta[C_t(d)-C_t(o)].
\]

Thus score similarity makes a rival available, but the tracker prior and observation-cost contrast determine whether it remains dynamically ambiguous. The current evidence rejects a score-only deterministic mapping from the first quantity to the second.

The event-level construct is reported as the ordered pair

\[
\mathcal A(e,t)=(\kappa(S_o,S_d), A_{dyn}(e,t)),
\]

not an arbitrary product or weighted sum. A scalar aggregation was not frozen or validated and must not be retrofitted to the opened outcomes.

## Paper-level contribution

The defensible contribution is:

> A decision-regret and trigger-coverage audit shows that event-conditioned delayed validation can pass a frozen symbolic lockbox yet fail in real audio because both the event distribution and the evidence boundary change.

The paper must not be reframed as a novel delayed-decision follower or a generally deployable recovery method.

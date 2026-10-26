# The use case: who to call to stop customers leaving

What this model is for, what it could be worth, and — most importantly — how we
measure whether it actually worked once it is live.

Every number here is reproduced by
[`notebooks/04_business_case.ipynb`](../notebooks/04_business_case.ipynb) from
the model's saved out-of-fold predictions (each customer scored by a model that
never saw them during training). Nothing is re-trained and nothing is quoted from
memory.

To try other assumptions than the ones below, the
[Churn Campaign Planner](https://claude.ai/artifact/Um1wxkVSasxiBG2831YXKa)
dashboard (private; ask for access) recomputes §3–§7 as you move the sliders.
At its default settings it shows the same numbers as this document.

## Summary

- **The problem:** customers who leave take **£10.9M of monthly revenue** with
  them — 28% of all revenue.
- **What the model does:** it sorts customers from most to least likely to leave.
  If we call the **top 10%** of that list, **83 out of every 100** people we call
  would otherwise leave.
- **Is calling worth it?** Calling costs money and only works sometimes. Under our
  assumptions (£5 per call, 1 in 10 would-be leavers persuaded to stay), calling
  the top 10% is worth about **£5M** over a year. It still breaks even if only
  **1 in 178** would-be leavers is persuaded to stay.
- **What we don't know:** how many people a call actually persuades. That is a
  property of our offer and our agents, not of the model, and no model can tell
  us.
- **How we find out:** each month, randomly **hold back 10% of the call list**
  and don't call them. Comparing that group with the people we called is the only
  reliable way to measure what the calls achieved. §7 is the step-by-step plan.

---

## 1. The decision the model supports

A retention team has a limited number of agent hours and a limited offer budget.
Each month it works down a call list, starting at the top, until it runs out of
time. **The model's only job is to put the right people at the top of that list.**

This is why the model was judged on **ROC-AUC** rather than accuracy:

- **Accuracy** asks "did we label each customer correctly as leaver or stayer?".
  It is easy to fool: 22.5% of customers leave, so a "model" that says *nobody
  leaves* is 77.5% accurate and is completely useless.
- **AUC** asks "if I pick one leaver and one stayer at random, how often does the
  model rank the leaver higher?". That is exactly what a call list needs. Our
  model's AUC is **0.916**: it puts the leaver higher about 92% of the time.

The API (`POST /predict`) returns a churn probability and a `risk_band`, where
`high` means the top 10% of customers. It deliberately does **not** say "call" or
"don't call" — that depends on budget, which is the campaign's decision, not the
model's. See [`serving.md`](serving.md).

## 2. How big is the problem, in money

Two terms used below:

- **MRR** (monthly recurring revenue) — what customers pay us each month.
- **ARPU** (average revenue per user) — MRR divided by the number of customers.

Measured on the 594,194 customers in the training data:

| | |
|---|---|
| Customers | 594,194 |
| Share of customers who leave | 22.5% |
| Share of **revenue** that leaves with them | **27.9%** |
| Total monthly revenue | £39.1M |
| **Monthly revenue of customers who leave** | **£10.9M** |
| Average monthly bill, leavers | £81.60 |
| Average monthly bill, stayers | £61.29 |

**Why the two percentages differ:** leavers pay on average £81.60 a month, a third
more than stayers (£61.29). So although they are 22.5% of customers, they are
27.9% of revenue. The people walking out are disproportionately the expensive
ones.

This matters for §6: when choosing whom to call, both "how likely are they to
leave?" and "how much do they pay?" count.

## 3. What the model's list gives us

If we sort all customers by the model's score and call the top X%:

| Call the top | Customers called | Of those called, % who would leave | % of all leavers we reach | Lift | % of leavers' revenue we reach |
|---|---|---|---|---|---|
| 1% | 5,941 | 95.8% | 4.3% | 4.3× | 4.6% |
| 5% | 29,709 | 89.6% | 19.9% | 4.0× | 21.4% |
| **10%** | **59,419** | **83.1%** | **36.9%** | **3.7×** | **39.2%** |
| 20% | 118,838 | 71.7% | 63.7% | 3.2× | 66.3% |
| 30% | 178,258 | 61.3% | 81.7% | 2.7× | 83.6% |
| 50% | 297,097 | 43.6% | 96.8% | 1.9× | 97.5% |

**How to read the 10% row:** we call 59,419 people. 83 in every 100 of them would
leave if we did nothing (if we called people at random, only 22–23 in 100 would).
Together they are 36.9% of all leavers and carry 39.2% of the revenue at risk.

**Lift** is that comparison as one number: 83.1% ÷ 22.5% = 3.7, i.e. the list
finds leavers 3.7 times as often as random calling.

![Cumulative gains and calibration](../reports/figures/04_value_at_risk.png)

## 4. Is it worth calling a given customer? The formula, built up

A call costs money, and it only pays off if (a) the customer was going to leave
and (b) the call changes their mind. So the value of calling one customer is:

```
value of calling  =  chance they would leave  ×  chance the call keeps them
                     ×  what keeping them is worth  −  cost of the call
```

In symbols, with one letter per piece:

| Symbol | Meaning | Where it comes from |
|---|---|---|
| `p` | chance this customer leaves if we do nothing | **the model** |
| `s` | share of would-be leavers that a call persuades to stay ("save rate") | **assumption: 10%** — unknown until we measure it (§7) |
| `m` | their monthly bill | **the data** (`MonthlyCharges`) |
| `H` | how many months a saved customer keeps paying | **assumption: 12 months** |
| `c` | cost of one call, including offer admin | **assumption: £5** |

So `m × H` is what a saved customer is worth, and `p × s` is the chance that
*this call* produces a save:

```
value of calling  =  p × s × m × H  −  c
```

**Call if this is above zero.** Rearranging ("above zero" means
`p × s × m × H > c`, then divide both sides by `s × H`) gives a simpler rule:

```
call if   p × m  >  c ÷ (s × H)   =   £5 ÷ (0.10 × 12)   =   4.17
```

`p × m` is simply **the monthly revenue we expect to lose from this customer**.
The rule says: call anyone whose expected monthly loss is above £4.17.

**Worked examples** (with `c = £5`, `s = 10%`, `H = 12`):

| Customer | `p` | `m` | Expected loss `p × m` | Value of calling `p × 0.1 × m × 12 − 5` | Call? |
|---|---|---|---|---|---|
| A — likely to leave, big bill | 0.80 | £90 | £72.00 | 0.8 × 0.1 × 90 × 12 − 5 = **+£81.40** | Yes |
| B — unlikely, big bill | 0.05 | £100 | £5.00 | 0.05 × 0.1 × 100 × 12 − 5 = **+£1.00** | Only just |
| C — fairly likely, small bill | 0.30 | £20 | £6.00 | 0.3 × 0.1 × 20 × 12 − 5 = **+£2.20** | Only just |
| D — unlikely, small bill | 0.10 | £20 | £2.00 | 0.1 × 0.1 × 20 × 12 − 5 = **−£2.60** | No |

**Why we can trust `p` as a real chance.** The formula multiplies `p` by money,
so it only works if "0.80" really means "8 in 10 such customers leave". AUC
doesn't check this — it only checks the *order*. So the notebook checks it
directly ("calibration"): across all customers the model predicts 22.52% will
leave and 22.52% actually do; among the top 10% it predicts 83.0% and 83.1%
actually do. The predictions match reality closely enough to be used as money.

### Why these three assumptions

They are not measured anywhere in this repo. They sit together in one notebook
cell so anyone can replace them and re-run.

- **`c = £5`** — a rough cost for an agent's time on one short outbound call plus
  admin. Replace with the real figure from the contact centre.
- **`H = 12 months`** — a deliberately simple, conservative horizon. A saved
  customer may stay longer; a real deployment would use expected lifetime.
- **`s = 10%`** — a placeholder. **This is the number that matters most, and it
  can only be measured by running the campaign (§7).** The good news: as §5
  shows, the go/no-go decision barely depends on it.

## 5. How far down the list to call

Using the rule above to order the list (by `p × m`, see §6) and the three
assumptions. **"Revenue kept" and "net" are 12-month totals** (because `H = 12`);
cost is the one-off cost of the calls.

| Call the top | Calls | Expected saves | Revenue kept (12 mo) | Net (12 mo) | Return per £1 spent | Cost per save | Value of the *last* call |
|---|---|---|---|---|---|---|---|
| 1% | 5,941 | ~550 | £644k | £615k | £21.7 | £54 | £99 |
| 5% | 29,709 | ~2,580 | £2.91M | £2.76M | £19.6 | £58 | £83 |
| **10%** | **59,419** | **~4,830** | **£5.29M** | **£4.99M** | **£17.8** | **£62** | **£68** |
| 20% | 118,838 | ~8,390 | £8.91M | £8.31M | £15.0 | £71 | £44 |
| 30% | 178,258 | ~10,750 | £11.1M | £10.2M | £12.5 | £83 | £22 |
| 50% | 297,097 | ~12,900 | £12.8M | £11.3M | £8.6 | £115 | £0.33 |

Column meanings:

- **Expected saves** = sum of `p × s` over everyone called.
- **Return per £1 spent** = revenue kept ÷ cost of calls.
- **Cost per save** = cost of calls ÷ expected saves.
- **Value of the last call** = the formula from §4 for the *last* person on the
  list. It tells you what one more call is worth.

![Campaign economics](../reports/figures/04_campaign_economics.png)

**Recommendation: call the top 10% or less, and spend more on the offer instead
of calling further down.** Total net value keeps rising all the way to 50%, which
looks like an argument to call half the customer base. But each extra call is
worth less and less: the last call at 1% is worth £99, at 30% only £22, and at
50% just £0.33. Agent hours are limited, so an hour spent on people near the top
is worth more than four hours spent at 30% depth.

**Break-even — why the decision doesn't hinge on the unknown `s`.** For the top
10%, the calls cost £297k. Each save is worth on average about £1,095 (12 months
of a ~£91 bill), so we need about **271 saves** to cover the cost. Among the
~48,300 people on the list who would actually leave, that is a save rate of
**0.56% — about 1 in 178 would-be leavers, or 1 save per ~220 calls.** We assumed
10%; even if the real rate is twenty times worse than that, the campaign still
pays for itself. The unknown `s` changes *how much* we make, not *whether* we
should run the campaign.

## 6. Sort the list by money at risk, not just by risk

The model ranks customers by `p` (chance of leaving). §4's rule says to rank by
`p × m` (expected monthly revenue lost).

**Example:** customer E has `p = 0.95` and a £20 bill (expected loss £19/month).
Customer F has `p = 0.70` and a £100 bill (expected loss £70/month). Ranking by
`p` calls E first. Ranking by `p × m` calls F first — F is less certain to
leave, but losing F costs more than three times as much.

Same 59,419 calls, two ways of choosing who gets them:

| Ranked by | % of those called who would leave | % of all leavers reached | Leavers' monthly revenue reached | Expected net value (12 mo) |
|---|---|---|---|---|
| chance of leaving `p` | 83.06% | 36.88% | £4,284,329 | £4,842,605 |
| **expected loss `p × m`** | 81.36% | 36.12% | **£4,410,050** | **£4,991,649** |
| difference | −1.7 points | −0.8 points | **+£125,721 / month** | **+£149,043** |

Ranking by expected loss reaches slightly *fewer* leavers but *more valuable*
ones: £126k more monthly revenue in reach and £149k more expected net value, for
the same number of calls. It costs nothing to do — the monthly bill is already
part of every request the API receives.

This is a good example of a model score and a business result disagreeing: the
`p × m` list scores worse on what the model was graded on and better on what the
campaign is for.

*Caveat:* the monthly bill stands in for customer value. A real deployment should
use profit margin rather than revenue, and an expected lifetime rather than a
flat 12 months. The rule stays the same; only the inputs improve.

## 7. Measuring results once we go live

Everything above is a **forecast**. None of it shows that calling anyone actually
stops them leaving. This section is the plan for measuring that, and it needs to
be in place **before the first campaign starts** — it cannot be added afterwards.

### Why we can't just look at the people we called

Suppose we call 1,000 high-risk customers and 800 of them leave anyway. Did the
calls work? We cannot tell: maybe 830 would have left without a call (calls
saved 30), or maybe 800 would have (calls did nothing). The model tells us who is
**at risk**, not who **can be persuaded** — a customer who has already decided to
leave and one who can be talked round look identical to it.

We need a group of similar customers who were *not* called, to show what would
have happened otherwise.

### The design: a random holdout

Each campaign month:

1. **Score** everyone and build the call list as in §5–§6 (e.g. top 10% by
   `p × m`).
2. **Randomly** set aside 10% of that list as the **holdout**. They are not
   called and get no offer. The other 90% are the **called group**.
   - It must be random (e.g. a seeded random number per customer), never chosen
     by an agent or by score. Otherwise the two groups differ for reasons other
     than the call, and the comparison is meaningless.
   - The holdout is drawn *from the call list*, so both groups have the same risk
     profile. It is not "the customers we didn't pick".
3. **Run the campaign** on the called group only.
4. **Wait for the measurement window** (fix it before launch, e.g. 90 days after
   the call date), then record who left.
5. **Compare** the leaving rates of the two groups (worked example below).

Customers stay in the group they were first assigned to. A holdout customer
should not be called next month just because they reappear on the list —
otherwise the holdout slowly disappears.

### What to record for every customer on the list

| Field | Why |
|---|---|
| customer ID | to join outcomes later. **The training data has no customer ID**, so this must come from the live system |
| campaign month, scoring date | to define the window |
| model version (bundle id) | to know which model produced the score |
| `p`, monthly bill, `p × m`, rank | to analyse results by risk level |
| group: `called` / `holdout` | the comparison |
| contacted? (reached / no answer / refused) | not everyone in the called group will be reached |
| offer made, offer cost | true cost per save |
| left within the window? (yes/no), date | the outcome |
| revenue over the window | to capture discounts and downgrades, not just leaving |

### How to read the result — a worked example

*The numbers below are made up to show the arithmetic; they are not predictions.*

Top-10% campaign, 59,419 customers, 10% holdout:

| | Customers | Left within 90 days | Leaving rate |
|---|---|---|---|
| Holdout (not called) | 5,941 | 4,931 | 83.0% |
| Called | 53,478 | 43,585 | 81.5% |
| **Difference** | | | **1.5 points** |

1. **Effect of the calls:** 83.0% − 81.5% = **1.5 percentage points fewer
   leavers** among those called.
2. **Is that real or luck?** The 95% margin of error for this difference is about
   ±1.0 points, so the true effect is likely between 0.5 and 2.5 points. That
   range is above zero, so the calls did something.
3. **Customers saved:** 1.5% × 53,478 called ≈ **800 customers**.
4. **Save rate `s`:** 1.5 ÷ 83.0 ≈ **1.8%** of would-be leavers persuaded. That is
   far below the 10% we assumed — so replace the 10% in the notebook with 1.8%
   and re-run §5.
5. **Money:** 800 saves × ~£91.70 × 12 months ≈ £880k kept, minus 53,478 × £5 ≈
   £267k cost ≈ **£613k net**. Positive, so keep going — and the margin of error
   says it was close to break-even (a 0.46-point drop), so the next step is to
   improve the offer, not to call more people.

The same comparison can be repeated on revenue over the window (average revenue
per customer, called vs holdout), which also captures the cost of discounts.

### How big the holdout needs to be

A smaller holdout costs less (fewer at-risk customers left uncalled) but can only
detect bigger effects. For a 59,419-customer campaign, at the usual standards
(95% confidence, 80% chance of detecting an effect that really exists):

| Holdout | Customers not called | Smallest drop in leaving rate we can reliably detect | Same, as save rate `s` |
|---|---|---|---|
| 5% | 2,970 | 2.1 points | 2.5% |
| **10%** | **5,941** | **1.5 points** | **1.8%** |
| 20% | 11,883 | 1.1 points | 1.4% |

**Recommendation: 10%.** If the real save rate is around 2% or more, one month
shows it clearly. If it is smaller than that, one month may come out
inconclusive; combine two or three months before deciding. Note that break-even
(a 0.46-point drop) is below what any sensible holdout can detect in one month:
the experiment can confirm "clearly worth it", but proving "only barely worth it"
would take several months of pooled data.

### What to do with the result

| Result | Action |
|---|---|
| Clear drop, `s` above ~2% | Keep running. Put the measured `s` into the notebook and re-check how deep to call |
| Drop not distinguishable from zero after 2–3 months | Calls aren't working as run. Change the offer or script (not the model), keep the holdout, measure again |
| Called group leaves *more* than holdout | Stop. Calls may be prompting people to leave |

### Keeping an eye on the model after launch

Once we act on the list, the called customers' outcomes are changed by our calls,
so they no longer show whether the model's predictions were right. **Check the
model on the holdout only**:

- **Is the list still finding leavers?** In the holdout, compare the predicted
  leaving rate (average `p`) with the actual rate. At launch they match (83.0%
  vs 83.1%). If they drift apart — say the model predicts 83% and only 70% leave —
  the model needs retraining.
- **Has the customer base changed?** `make drift` compares incoming customers
  with the training data.
- **Is the service healthy?** `/metrics` exposes request counts
  (`churn_predictions_total`), speed (`churn_prediction_latency_seconds`), the
  spread of scores (`churn_predicted_probability`) and inputs outside the
  training range (`churn_out_of_range_inputs_total`), for Prometheus.

### The four levels of measurement

| Level | Question | Measured by | Status |
|---|---|---|---|
| Model | Does it rank risk well? | AUC, lift, calibration | **Done** — `experiments.csv`, notebook 04. Folds are frozen (fingerprint `13feb807`); `tests/test_cv.py` fails if they change |
| Decision | Is a campaign worth running? | Net value, cost per save, break-even | **Done** — notebook 04 §4–5. Re-run when `c`, `s` or `H` change |
| **Effect** | **Did calling actually help?** | **Holdout comparison above** | **Not built — needs the live campaign** |
| System | Is the service still working? | Prometheus metrics, `make drift` | **Done** |

The missing level is the one that matters most. A better model would add
+0.0002 AUC, which is smaller than this setup's measurement noise
([`learnings.md`](../learnings.md)) and would change none of the numbers above. A
measured `s` changes all of them. **The most valuable next step is the holdout,
not a better model.**

### Launch checklist

- [ ] Customer IDs available at scoring time and stored with each score
- [ ] Random holdout assignment implemented, seeded and logged
- [ ] Measurement window (e.g. 90 days) and the definition of "left" agreed
- [ ] Holdout customers excluded from all retention contact, not just this campaign
- [ ] Real values for `c` and `H` in place of the assumptions
- [ ] Results table (fields above) set up before the first call

## 8. Limits

- **The data is synthetic** — a generated expansion of the public Telco dataset.
  The £10.9M and everything derived from it describe that generator, not a real
  operator. The method transfers; the numbers don't.
- **£5, 10% and 12 months are assumptions**, not findings.
- **Nothing here proves cause and effect.** For example, month-to-month contracts
  are the strongest risk signal ([notebook 99](../notebooks/99_final_writeup.ipynb)
  §5), but moving someone to an annual contract won't necessarily make them stay —
  people who already plan to stay are the ones who choose annual contracts. Only
  the holdout measures cause and effect.
- **No contact history.** The model doesn't know if someone was called three times
  this quarter. The live system needs to suppress repeat calls itself.
- **The call/no-call decision belongs to the campaign, not the API.** The API keeps
  returning a probability and a risk band; see [`serving.md`](serving.md).

---

## Reproducing every number here

```bash
make data                                    # Kaggle CSVs into data/raw/
uv run jupyter lab notebooks/04_business_case.ipynb
```

The notebook first checks that the served model's predictions are **bitwise
identical** to champion run `0904_202729_lgbm`, so the numbers describe the model
that scored public LB 0.91393, not a re-fit that resembles it.

The dashboard's data comes from the same inputs and makes the same check:

```bash
uv run python -m scripts.build_dashboard_data   # -> reports/dashboard/*.csv
```

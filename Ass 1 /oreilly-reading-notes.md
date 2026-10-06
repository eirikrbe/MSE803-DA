# O'Reilly reading notes for MSE803 Assessment 1

Pulled on 2 Oct 2026 from O'Reilly Learning, using your account. These are **paraphrased notes and pointers**.
They are not text to paste into the assessment. Read the named section yourself before you cite it,
and choose any direct quote yourself. O'Reilly's online editions have no page numbers, so in APA 7
you cite the chapter and section name instead, e.g. *(Bruce et al., 2020, Chapter 3, "Multiple Testing" section)*.

`→` marks how a point could feed your answer. Whether to use it is your call.

---

## Task 1-B: techniques (water quality ↔ fish health)

**Bruce, Bruce & Gedeck (2020), Ch 5, "Logistic Regression" incl. "Generalized Linear Models"**
- Logistic regression is a GLM for a binary outcome such as fish present/absent. It is fitted by maximum likelihood,
  not least squares, so it has no R² or RMSE and you judge it with classification metrics.
- The coefficients are on the log-odds scale. Exponentiating one gives an odds ratio, e.g. "each extra mg/L of DO
  multiplies the odds of native fish being present by X". The authors list ease of interpretation as the method's
  main advantage over other classifiers.
- A GLM is a distribution family plus a link function. Poisson and negative binomial are the families used for
  **counts**. The authors warn that these need more care than logistic regression, so if you propose them, say so.
- Spline terms and GAMs also work inside logistic regression. That gives you a threshold-shaped DO effect without
  having to guess a polynomial degree.
- In these models a p-value is only a rough indicator of which variables matter, not a formal test.
- → Backs logistic regression over SVM for a board audience (technique 2), and the count-model upgrade of technique 1.

**Same book, Ch 5, "Evaluating Classification Models" → "The Rare Class Problem"**
- When one class is rare, a model that always predicts the common class can score high accuracy and still be
  useless. Use recall, precision and specificity, ROC/AUC, or precision–recall curves instead. Lower the decision
  cut-off when missing the rare class costs more than a false alarm.
- → This applies only once the organisation records zero counts. The Oct–Dec 2023 file has none, so
  presence/absence can't be modelled yet. When it can, report recall on "absent", not overall accuracy
  (your W5 majority-class baseline habit).

**Same book, Ch 4 (Regression and Prediction)**
- *Prediction Versus Explanation*: a fitted regression does not prove which way causation runs. That has to come
  from domain knowledge. → Goes in your "what this does not show" paragraph.
- *Confounding Variables*: leaving out an important variable makes the coefficients you did fit misleading.
  - The Avon candidates are season, flow, sampling time of day and habitat.
- *Correlated Predictors*: when inputs are correlated, coefficient signs become hard to read.
  - Temperature and DO are physically linked.
  - → This is another justification for using % saturation, and for recording sampling time.
- *The Dangers of Extrapolation*: a model only holds within the range of data it was fitted on.
  → October–December readings cannot tell you about the summer DO lows.
- *Cross-Validation*: in-sample metrics versus k-fold holdout. You already use LOOCV, so this connects your W3 work.
- *Generalized Additive Models*: fit splines automatically (pyGAM in Python).

**Croll & Yoskovitz (2013), Lean Analytics, Ch 4: the "How to Think Like a Data Scientist" list of ten data pitfalls**
- **Ignoring seasonality** → include seasonal terms in 1-B.
- **Alerts that are too sensitive end up being ignored** → design the alert threshold carefully in 1-D.
- **Combining your data with outside sources** (e.g. rainfall, stormwater outfalls, council monitoring)
  → a data-integration step in 1-B.

**Provost & Fawcett (2013), Appendix A, "Proposal Review Guide"**
- It is a checklist of questions:
  - Is the target variable defined precisely?
  - Do the training data come from the same population the model will be applied to?
  - Does the model type fit what you already know (e.g. a linear model for a clearly nonlinear problem)?
  - Will domain experts need to check the model, and can they understand it?
  - Which baselines will you compare against?
  - Is the evaluation done on holdout data?
- → Use it to structure your "key steps" so they visibly cover target, data prep, model choice, evaluation and baseline.
  It works for Task 2-A as well (evaluating each project as a proposal).

---

## Task 1-C: tools and visualisations

**Wilke (2019), Ch 28, "Choosing the Right Visualization Software"**
- Judge a tool on three things:
  - **reproducibility/repeatability**
  - **speed of exploration**
  - **control over design**
- Interactive tools often don't record the transformations behind a figure, which makes it hard to redo later.
  Figures made by a script can be regenerated whenever new data arrive.
- Exploration and presentation are different phases, and it is reasonable to use different tools for each.
- → A principled case for a two-tool pairing:
  - Python as the scripted, repeatable analysis you rerun each season.
  - A BI tool as the presentation layer for staff and the board.

**Knaflic (2015), Ch 1, "The importance of context"**
- *Exploratory vs explanatory*: show the audience the few findings that matter, not every angle you explored.
- *Who / What / How*: name a specific audience, decide what they should know or **do**, and only then pick the
  data that supports it. The board and the public may need different products.
- Don't leave out data that cuts against your point.
- *Big Idea* (from Duarte): a single sentence that states your point of view and what is at stake.
- → Frame each visual by its audience and the action you want (e.g. the board approves AV-3 restoration funding).

**Knaflic (2015), Ch 2, "Choosing an effective visual"**
- If you only have one or two numbers, show the numbers themselves, large ("simple text").
- Use line graphs for time, and keep the time intervals consistent. Your sampling dates are irregular, so use a
  real date axis.
- A *slopegraph* suits two time points: emphasise one series and grey the rest. → e.g. a site before and after restoration.
- Bar charts need a zero baseline. Line graphs can use a non-zero baseline if you do it carefully.
- Avoid pie charts, 3D and **secondary y-axes**. Instead, stack panels that share the x-axis.
  - Plotting two series against one axis can suggest a relationship that may not exist.
  - → Show DO and fish as two aligned panels, never as a dual-axis chart.
- Test the chart on a colleague: where do their eyes go first?

**Knaflic (2015), Ch 4, "Focus your audience's attention"**
- Preattentive attributes (colour, size, position) direct attention within the first few seconds.
- Set everything to grey, then highlight the one element that matters.
- Use colour sparingly and consistently.
- Make it **colour-blind safe**: roughly 8% of men are affected. Avoid red/green pairs; blue/orange is safer.
- Put the most important element at the top left, since readers scan in a Z pattern.

**Wilke (2019), Ch 19, "Common Pitfalls of Color Use", and Ch 20, "Redundant Coding"**
- Qualitative colour scales work for about 3–5 categories. Beyond about 8, label directly. Avoid rainbow scales.
- Red–green contrasts disappear for the most common colour-vision deficiency, so a red/amber/green traffic light
  **on its own** is risky.
- Encode the same information more than one way (colour + text label + icon).
- Label directly rather than using a legend. Where a legend is needed, match its order to the visual order
  (here, river order).
- Test the figure in a colour-blindness simulator. (Wilke happens to use the Iris data as his example, the same
  dataset as your W1.)
- → For the river-check grid (Figure C): band letter + word + colour, e.g. "C · moderate stress".

**Wilke (2019), Ch 29, "Telling a Story and Making a Point"**
- A story arc runs opening → challenge → action → resolution. One figure rarely carries the whole arc.
- *Make a figure for the generals*: busy decision-makers need one clear point, so drop dimensions that don't serve it.
- Build up from simple figures to complex ones. Start close to the raw data, then show derived quantities.
- *Isotype* plots (repeated pictograms whose count encodes the data) make figures more memorable. The pictograms
  must encode data, not decorate the chart.
- → For a public-facing version, fish icons could encode counts. Show raw values first, then the derived ones
  (% saturation, the band).

---

## Task 1-D: recommendations

- **Lean Analytics, Ch 2, "What Makes a Good Metric?"**
  - A good metric is comparative, understandable, a ratio or rate, and it **changes behaviour**.
  - Agree in advance what you will do for each possible result.
  - → For each recommendation, write down what reading triggers action and what result would show the fix worked.
- **Lean Analytics, Ch 4 (pitfalls list)**
  - Alerts that fire too often get ignored.
  - → Anchor the DO alert to a defensible standard (NPS-FM/NIWA), and plan to review false alarms.
- **Knaflic, Ch 1**: the person who analysed the data should make specific recommendations, not just present the data.

---

## Task 2-A: strengths and weaknesses, data types

**Provost & Fawcett (2013), Ch 1**
- **Data-driven decision-making (DDD)** means deciding on the basis of data analysis rather than intuition alone.
  It is a matter of degree, not all or nothing.
- **Evidence for DDD:** Brynjolfsson, Hitt & Kim (2011) found that firms one standard deviation more data-driven were
  roughly 4–6% more productive. Cite the original study if you use this.
- **Two kinds of data-driven decision:**
  - discovering something new in the data;
  - **repeated decisions at scale**, where small accuracy gains add up.
  - → Project A's personalisation is the second kind. Project B's tests are mostly the first.
- **Strategic assets:** data and data-science capability are complementary, and both need investment.
- **Signet Bank / Capital One:**
  - The bank's historical data only covered the customers and terms it had already offered, so it could not model new
    offers.
  - It ran **randomised offers**, accepting losses, to buy that data, and only then built its models.
  - → Strong evidence that experimentation (B's capability) generates the unbiased data that predictive
    personalisation (A) needs.
- **Martens & Provost (2011):** prediction kept improving as more fine-grained behavioural data were added. So A's
  upside depends on how much detailed data the company has. That is a strength if it has the data, a weakness if not.
- **Target's pregnancy prediction** (Duhigg, 2012) raised ethical concerns.

**Same book, Ch 14, "What Data Can't Do: Humans in the Loop, Revisited" and "Privacy, Ethics, and Mining Data About Individuals"**
- **Choosing the objective:** only people can decide what to optimise, and the true objective often can't be
  measured. Models then optimise a proxy, which is a risk once deployed.
- **Data are not neutral:** data carry the choices and biases of whoever designed the collection, so you need to know
  which population was sampled. The chapter's "cellsite zero" story is a good example of a data leak.
- **One-off strategic decisions** may have too little data behind them, so experienced judgement is still needed.
  → In 2-B, leadership's decision should draw on both the data and broader strategy.
- **Privacy vs effectiveness:**
  - Goldfarb & Tucker (2011) found that ad effectiveness fell sharply after EU privacy rules came in.
  - → Regulation directly erodes the value of A.
  - Privacy itself is hard to define (Solove 2006; Nissenbaum's *Privacy in Context*).

**Bruce et al. (2020), Ch 3, "Statistical Experiments and Significance Testing"**
- **Randomisation:** a randomised control group is what lets you attribute a difference to the change rather than to
  something else. → This is B's core strength.
- **Choose the metric before the test.** Choosing it afterwards opens the door to researcher bias.
- **Power and sample size:**
  - Sample size, effect size, alpha and power are linked: fix three and the fourth follows.
  - Their worked example needs **almost 120,000 impressions** to detect a 10% lift on a ~1.1% click-through rate at
    80% power, compared with about 5,500 for a 50% lift.
  - → A weakness of B for low-traffic features.
- **Multiple testing:**
  - Run 20 tests at α = 0.05 and there is about a 64% chance of at least one false positive.
  - Bayer could fully replicate only 14 of 67 studies it re-ran.
  - → A pitfall for B, and a reason A's models need holdout validation.
- **p-values:** the ASA's 2016 statement says decisions shouldn't rest on a p-value threshold alone, and statistical
  significance is not the same as practical importance.
- **Timing:** visitors differ by time of day, day of week, season and device, so run tests over complete cycles.
- **Multi-arm bandits** move traffic towards the better variant while the test is still running. They are also a step
  towards personalisation.
- **Uplift** (Ch 5, sidebar under "Lift"): a model trained on A/B-test results can predict which treatment works best
  for which customer. → That is personalisation built directly on B's experiments.

**Croll & Yoskovitz (2013), Lean Analytics, Ch 2, "How to Keep Score"**
- **Qualitative vs quantitative:** qualitative data answer "why", quantitative data answer "what" and "how much".
  → B needs both A/B results and user feedback.
- **Vanity vs actionable metrics.**
- **Leading vs lagging:** churn lags, complaints lead. → A's CRM predictions depend on leading indicators.
- **Correlation vs causation:** causation is established through controlled experiments.
  → A's models are mostly correlational; B's tests are causal.
- **Traffic limits:** most firms have more ideas to test than traffic to test them with.

---

## Task 2-B: pitfalls, mitigations, recommendation

**Project A (AI CRM)**
- **Optimisation without human judgement (Lean Analytics Ch 4):** Orbitz showed pricier hotels to Mac users and got
  bad press. → An unintended-consequences pitfall for personalisation.
- **Data pitfalls in recommenders (Lean Analytics Ch 4 pitfalls list):**
  - If heavy "superfan" users stay in the data, everyone gets recommended the same items (popularity bias).
  - Never assume the data are clean.
- **Real penalties (Practical Fairness Ch 1):**
  - GDPR fines can reach 4% of global revenue.
  - The FTC fined Facebook $5bn in 2019.
  - HUD sued Facebook over housing-ad targeting linked to protected characteristics.
- **The same chapter's "rules to code by":**
  - Re-check systems over time so they don't become self-perpetuating.
  - Users, not the company, decide what is private.
  - Test: would this be acceptable if a person did exactly the same thing?
  - Ask whether you are building it only because competitors are.
- **Uneven performance (Practical Fairness Ch 11):** products can work worse for under-represented groups (voice
  recognition, optical heart-rate sensors).
  - → Audit personalisation performance by customer segment.
  - The chapter ends with a fair-products checklist covering accountability, autonomy, transparency and proof.

**Project B (A/B testing + feedback)**
- **Data-informed, not data-driven (Lean Analytics Ch 4):**
  - Optimisation can only find a **local maximum**: it can perfect a tricycle but will never suggest a fourth wheel.
  - Omniture's click-optimised imagery hurt the client's brand. → Metric myopia.
  - The chapter's idea is that people generate ideas and machines test them.
- **False metrics (Lean Analytics Ch 2):**
  - Staff gamed customer-satisfaction ratings.
  - Rewarding pipeline counts produced junk leads.
  - → Goodhart-style gaming. This applies to A too, if sales staff learn to game AI lead scores.
- **Dark patterns (Practical Fairness Ch 11):**
  - Mathur et al. found 1,818 dark patterns across 11k shopping sites, in seven categories: sneaking, urgency,
    misdirection, social proof, scarcity, obstruction and forced action.
  - → This is what conversion-optimised testing can drift into. Use the categories as a review checklist.
- **Experiment ethics and method (PSDS Ch 3):**
  - Facebook's 2014 emotional-contagion experiment ran without users' knowledge.
  - Also from this chapter: multiple testing, power, and fixing the metric in advance.
- **Limits of feedback (DSfB Ch 14):** Steve Jobs on focus groups — users can't always tell you what they want.

**Recommendation (sequencing, the "different approach" option)**
- **Data-science maturity (DSfB Ch 13, "A Firm's Data Science Maturity"):**
  - Immature firms rarely deploy sophisticated solutions at scale.
  - Mid-maturity firms have a framework for testing alternatives.
  - Mature firms measure the incremental effect of their actions inside an experimentation framework.
  - The chapter also says a workable solution now can beat a better one next year.
  - → B builds the maturity that A depends on.
- **Competitive advantage (DSfB Ch 13, "Achieving Competitive Advantage with Data Science"):**
  - A data asset only gives an advantage if it fits the firm's strategy and competitors can't easily copy it
    (Dell vs Compaq, Amazon vs Borders).
  - → Ask whether this company has unique CRM data or strategic fit for A.
- **Experimentation as data acquisition (DSfB Ch 1, Signet / Capital One):** see Task 2-A above.
- **AI Hierarchy of Needs (Rogati, 2017, Hackernoon; not on O'Reilly):** puts experimentation below ML/AI in the pyramid.

---

## References (APA 7th; check against Yoobee's referencing guide)

Bruce, P., Bruce, A., & Gedeck, P. (2020). *Practical statistics for data scientists: 50+ essential concepts using R and Python* (2nd ed.). O'Reilly Media.

Croll, A., & Yoskovitz, B. (2013). *Lean analytics: Use data to build a better startup faster*. O'Reilly Media.
*(The copyright page says © 2013 first edition. O'Reilly's 2024 listing is a softcover reprint of the same edition, so cite 2013.)*

Knaflic, C. N. (2015). *Storytelling with data: A data visualization guide for business professionals*. Wiley.

Nielsen, A. (2020). *Practical fairness: Achieving fair and secure data models*. O'Reilly Media.

Provost, F., & Fawcett, T. (2013). *Data science for business: What you need to know about data mining and data-analytic thinking*. O'Reilly Media.

Wilke, C. O. (2019). *Fundamentals of data visualization: A primer on making informative and compelling figures*. O'Reilly Media.

**Primary studies the books point to.** If you use one, look it up and cite it directly:

- Brynjolfsson, E., Hitt, L. M., & Kim, H. H. (2011). *Strength in numbers: How does data-driven decisionmaking affect firm performance?* SSRN working paper.
- Duhigg, C. (2012, February 16). How companies learn your secrets. *The New York Times Magazine*.
- Goldfarb, A., & Tucker, C. E. (2011). Privacy regulation and online advertising. *Management Science, 57*(1), 57–71.
- Haroz, S., Kosara, R., & Franconeri, S. L. (2015). ISOTYPE visualization: Working memory, performance, and engagement with pictographs. *Proceedings of CHI 2015*.
- Kramer, A. D. I., Guillory, J. E., & Hancock, J. T. (2014). Experimental evidence of massive-scale emotional contagion through social networks. *PNAS, 111*(24), 8788–8790.
- Mathur, A., et al. (2019). Dark patterns at scale: Findings from a crawl of 11K shopping websites. *Proceedings of the ACM on Human-Computer Interaction, 3*(CSCW).
- Wasserstein, R. L., & Lazar, N. A. (2016). The ASA statement on p-values: Context, process, and purpose. *The American Statistician, 70*(2), 129–133.

---

**Sections read:**
- PSDS:
  - Ch 3: all of it.
  - Ch 4: "Prediction Versus Explanation", "Cross-Validation", "The Dangers of Extrapolation", "Correlated Predictors", "Confounding Variables", "Generalized Additive Models".
  - Ch 5: "Logistic Regression" and "Evaluating Classification Models".
- DSfB: Ch 1; Ch 13 ("Achieving Competitive Advantage", "A Firm's Data Science Maturity"); Ch 14 ("What Data Can't Do", "Privacy, Ethics…"); Appendix A.
- Lean Analytics: Ch 2 (most sections) and Ch 4.
- Storytelling with Data: Ch 1, Ch 2, Ch 4.
- Wilke: Ch 19, 20, 28, 29.
- Practical Fairness: Ch 1 (from "What If I'm Skeptical…" onwards) and Ch 11 ("Products That Work Better…", "Dark Patterns", "Fair Products Checklist").

---

## Added on 2 Oct 2026 (second pass)

**Provost & Fawcett (2013), Ch 7: "A Key Analytical Framework: Expected Value" and "Evaluation, Baseline Performance, and Implications for Investments in Data"** (Task 2-B; also 1-B)
- **Expected value** = sum over outcomes of probability × value. The **probabilities come from data**, but the
  **values, costs and benefits usually come from business knowledge**, often only as ranges.
  - → A clean way to frame "A or B?": the likelihood of each project paying off, times its value, minus its cost.
  - Say where each number would come from.
- **Choosing the threshold.** Costs and benefits set the decision threshold, not a default of 50%. In the authors'
  example, it pays to target anyone with more than a 1% chance of responding.
  - → For Task 1-B: set the classifier's cut-off from the cost of missing a fish-kill versus the cost of a false alarm.
- **Baselines** (the section on baseline performance):
  - Compare against something *simple but not simplistic*: the majority class, the mean, a one-feature model, or
    "received wisdom". The weather example: forecasts must beat both "same as today" and "the long-run average".
  - Compare a model *with and without* each data source, to justify what that data costs.
  - → For Project A: does the AI model beat simple segmentation rules?
- **Two traps in cost/benefit tables:** inconsistent signs, and double-counting the same benefit or cost.

**Nielsen (2020), Ch 9, "ML Models and Privacy"** (Project A pitfalls and mitigations)
- **Privacy keeps moving,** technically and legally: data once thought anonymous can later be re-identified.
  Take reasonable precautions for today and expect to revise them.
- **Deletion is hard once data has trained a model.** The GDPR's right to deletion (Art. 17) is difficult to
  honour for data already used in ML training.
- **Models can leak personal information.** ML can infer private facts from observable traits.
- **Attacks on models themselves:**
  - membership inference (was this person in the training data?);
  - data leaks through overfit models that memorise records;
  - model inversion (reconstructing training inputs);
  - model extraction (copying the model's logic).
- **Mitigations:**
  - **differential privacy** for training and reporting, with a "privacy budget";
  - **k-anonymity** for any dataset you release;
  - test models for these attacks before release.

**Wilke (2019), Ch 15, "Visualizing Geospatial Data"** (Task 1-C, the river map)
- Maps are intuitive but easy to get wrong. Keep only the layers that serve the point, and add a scale bar and a north arrow.
- Points on a map can carry extra variables through colour or shape (e.g. sites coloured by oxygen band, with the
  shape showing whether native fish were present).
- On a light background, darker reads as "more". **Binned colour scales (about 4–6 bins) are easier to read than
  continuous ones.** → This supports using the A–D bands.
- **Area bias.** Large areas look like large amounts. A *cartogram heatmap* (equal-size tiles) treats every unit
  equally. → This is a principled reason for the river-check grid's equal tiles in a fixed order, rather than a
  geographic map where long reaches would dominate.

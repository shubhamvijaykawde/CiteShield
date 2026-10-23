# LLM judge calibration

Judge `qwen3:4b`, answer prompt `ac2a420f`, refusal prompt `46212112`. Labels: 60 of 60 sample items, by one annotator (the project author). Split by golden question; the held-out half decides the verdicts.

This is the frozen judge (eval/labels/frozen_judge.json). Its prompt was revised on the dev split, so dev agreement is optimistic; held-out is the independent measure.

Target: kappa >= 0.7 with the lower end of the bootstrap 95% interval also >= 0.7, on at least 20 held-out pairs with at least 5 human labels in each of two classes.

| grade | split | n | kappa | 95% CI | agreement | human classes | verdict |
|---|---|---|---|---|---|---|---|
| correctness | dev | 24 | 0.541 (unweighted 0.375) | 0.24 to 0.77 | 0.667 | {'1': 4, '2': 15, '3': 5} |  |
| correctness | holdout | 24 | 0.694 (unweighted 0.566) | 0.43 to 0.89 | 0.750 | {'1': 3, '2': 11, '3': 10} | below target |
| faithful | dev | 24 | 0.676 | 0.38 to 0.92 | 0.833 | {'False': 14, 'True': 10} |  |
| faithful | holdout | 24 | 0.391 | -0.01 to 0.74 | 0.708 | {'False': 10, 'True': 14} | below target |
| citation_adequate | dev | 24 | 0.565 | 0.18 to 0.90 | 0.792 | {'False': 9, 'True': 15} |  |
| citation_adequate | holdout | 24 | 0.500 | 0.14 to 0.82 | 0.792 | {'False': 9, 'True': 15} | below target |
| refusal_appropriate | dev | 6 | 1.000 | 1.00 to 1.00 | 1.000 | {'False': 3, 'True': 3} |  |
| refusal_appropriate | holdout | 6 | 1.000 | 1.00 to 1.00 | 1.000 | {'False': 3, 'True': 3} | insufficient data |

Held-out agreement by origin (synthetic = deliberately perturbed answers):

| grade | real n | real agreement | synthetic n | synthetic agreement |
|---|---|---|---|---|
| correctness | 18 | 0.722 | 6 | 0.833 |
| faithful | 18 | 0.667 | 6 | 0.833 |
| citation_adequate | 18 | 0.833 | 6 | 0.667 |
| refusal_appropriate | 6 | 1.000 | 0 | n/a |

Held-out confusion, correctness:

| human \ judge | 1 | 2 | 3 |
|---|---|---|---|
| 1 | 1 | 2 | 0 |
| 2 | 0 | 8 | 3 |
| 3 | 0 | 1 | 9 |

Held-out confusion, faithful:

| human \ judge | False | True |
|---|---|---|
| False | 6 | 4 |
| True | 3 | 11 |

Held-out confusion, citation_adequate:

| human \ judge | False | True |
|---|---|---|
| False | 4 | 5 |
| True | 0 | 15 |

Held-out confusion, refusal_appropriate:

| human \ judge | False | True |
|---|---|---|
| False | 3 | 0 |
| True | 0 | 3 |

Held-out disagreements: 9 (not listed, to keep them out of prompt iteration).

Dev disagreements (12):

- q001-3ef00cccede0 (real, answered): human {'correctness': 2, 'faithful': False, 'citation_adequate': True}, judge {'correctness': 2, 'faithful': True, 'citation_adequate': True}. Judge: correctness: 2 (omits consent and legitimate interests basis with the exception for public authorities); faithful: true (all statements in the summary and claims are supported by the cited source text Note: The answer omits consent under Article 6(1)(a) and legitimate interests under Article 6(1)(f), incorrectly stating that there are only four lawful bases. Article 6(1) is nevertheless the appropriate citation.
- q004-89c83263dc34 (real, answered): human {'correctness': 2, 'faithful': True, 'citation_adequate': True}, judge {'correctness': 2, 'faithful': True, 'citation_adequate': False}. Judge: omits controller's obligation to demonstrate consent and pre-consent information requirement; cites Art. 7(3) for withdrawal effect but not for other conditions Note: The withdrawal claim is accurate and supported by Article 7(3), but the answer omits the controller's duty to demonstrate consent and the requirements to inform the data subject beforehand and make withdrawal as easy as giving consent.
- q007-4e1149913171 (real, answered): human {'correctness': 2, 'faithful': True, 'citation_adequate': True}, judge {'correctness': 3, 'faithful': True, 'citation_adequate': True}. Judge: Covers all three cases from reference answer with correct qualifiers; minor wording differences (e.g., 'significant impacts' for 'similarly significantly affect', omission of 'offences' in claim 2) do Note: The three specific cases are accurately described and cited under GDPR Article 35(3). However, the answer omits the broader Article 35(1) requirement for processing likely to result in high risk, particularly when using new technologies.
- q026-812ce1c0d1ad (real, answered): human {'correctness': 2, 'faithful': False, 'citation_adequate': True}, judge {'correctness': 3, 'faithful': True, 'citation_adequate': True}. Judge: The answer covers all key points from the reference answer with their qualifiers: (1) data governance practices covering design choices, data collection origins, data preparation, bias assessment, mit Note: The answer omits assumptions, assessment of data availability/quantity/suitability, and completeness. It also says datasets must be "free of errors" without the source's qualifier "to the best extent possible." Articles 10(2), 10(3), and 10(4) are appropriate citations.
- q031-2e05dc22b524 (real, answered): human {'correctness': 3, 'faithful': False, 'citation_adequate': True}, judge {'correctness': 2, 'faithful': False, 'citation_adequate': False}. Judge: The answer adds 'cooperate with authorities' as an obligation, but this is not part of the reference answer and is not covered by the cited sources (AI Act Art. 53(1) does not include a general cooper Note: The summary covers all four key obligations in the reference. However, "cooperate with authorities" is broader than the specific duty to provide technical documentation to the AI Office and national competent authorities upon request.
- q031-cdf5c90b1feb (synthetic, perturbed_contradiction): human {'correctness': 1, 'faithful': False, 'citation_adequate': False}, judge {'correctness': 2, 'faithful': False, 'citation_adequate': False}. Judge: correctness: 2 (omits the obligation to share information with integrators, the copyright compliance policy, and the training data summary); faithful: false (the claim states "which shall not contain, Note: The summary contains all four key obligations in the reference. However, “cooperate with authorities” is broader than the specific obligation in the cited text to provide technical documentation to the AI Office and national competent authorities upon request. Corrected on review: claim 1 says "shall not contain", contradicting the cited Art. 53(1)(a) ("shall contain").
- q040-1214750557bc (real, answered): human {'correctness': 2, 'faithful': False, 'citation_adequate': True}, judge {'correctness': 2, 'faithful': False, 'citation_adequate': False}. Judge: The reference answer requires two key conditions: (1) average monthly active recipients in the Union ≥45 million; (2) Commission designation as very large platform. The answer states only condition (1 Note: The 45-million threshold is correct, but designation also requires the Commission to designate the platform under Article 33(4). Article 33(1) is the appropriate citation, but the answer omits this separate condition.
- q053-345326bb9ab2 (real, answered): human {'correctness': 1, 'faithful': False, 'citation_adequate': False}, judge {'correctness': 2, 'faithful': True, 'citation_adequate': False}. Judge: omits the NIS2 deadlines entirely; citation_adequate false because one regulation (NIS2) is not cited at all Note: The answer addresses only the GDPR deadline and omits the NIS2 reporting stages: early warning within 24 hours, incident notification within 72 hours, and final report within one month. It also omits the GDPR qualifications "where feasible" and the exception for breaches unlikely to pose a risk. GDPR Article 33(1) supports the GDPR claim, but the required comparison and NIS2 citations are missing.
- q079-ba900e91f2ef (synthetic, perturbed_wrong_citation): human {'correctness': 1, 'faithful': False, 'citation_adequate': False}, judge {'correctness': 2, 'faithful': False, 'citation_adequate': False}. Judge: correctness: 2 (omits the data subjects part, a key point of the reference answer); faithfulness: false (the claim cites Art. 34(4) for the obligation but Art. 34(4) does not state the controller's ob Note: The answer omits the comparison with communicating the breach to affected data subjects. The 72-hour supervisory-authority notification rule is not supported by the cited Article 34(4); Article 33(1) is the appropriate citation for that rule, while Article 34(1) addresses communication to data subjects when a breach is likely to result in high risk.
- q079-e30b669b75cb (real, answered): human {'correctness': 2, 'faithful': False, 'citation_adequate': False}, judge {'correctness': 2, 'faithful': True, 'citation_adequate': True}. Judge: The reference answer specifies that authority notification is mandatory within 72 hours unless the breach is unlikely to pose risk. The answer states the obligation and timing for authority notificati Note: The answer omits the Article 33(1) exception for breaches unlikely to pose a risk. It also fails to explain the controller's duty under Article 34(1) to communicate a breach to data subjects without undue delay when high risk is likely. Article 34(4) supports the supervisory authority's power to require communication, but does not replace Article 34(1) as the citation for the underlying duty.
- q087-7d031f150b1c (real, answered): human {'correctness': 2, 'faithful': False, 'citation_adequate': False}, judge {'correctness': 3, 'faithful': False, 'citation_adequate': True}. Judge: correctness: covers all 11 reference points (the reference point 4 is split into two claims but covered by claims 4 and 5); faithful: the summary states the obligations apply to providers, distributor Note: The checklist correctly covers the 12 provider obligations in AI Act Article 16. However, the summary incorrectly states that these obligations apply to distributors and deployers too. Article 16, as cited, establishes the obligations for providers of high-risk AI systems.
- q097-47e50abea633 (real, answered): human {'correctness': 2, 'faithful': True, 'citation_adequate': True}, judge {'correctness': 3, 'faithful': True, 'citation_adequate': True}. Judge: The answer correctly states the high-risk status with the exception for financial fraud detection as per Annex III point 5(b), matching the reference answer without dropping any qualifiers or adding e Note: The category and financial-fraud exception are correctly reproduced from Annex III, but the answer should explicitly state that these systems are classified as high-risk.

Limits: one annotator, no inter-annotator agreement; a small sample, so the intervals are wide; synthetic perturbations are simpler than real failures.

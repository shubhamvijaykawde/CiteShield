# CiteShield annotation packet: held-out items

30 items (24 answer, 6 refusal), in the order `citeshield-label` presents them. Each item block below is exactly what the labelling CLI shows: the item id and kind, then the question, reference answer, system answer and cited sources. Grade only from what is in the block.

## Rubric (verbatim from eval/labels/GUIDE.md)

## Answer items: three grades

Grade them independently: an answer can be correct but unfaithful to its sources, or
faithful but citing the wrong provision.

### correctness (1, 2 or 3), against the reference answer

| grade | meaning |
|---|---|
| 3 | Every key point of the reference is present and nothing contradicts it. Extra correct detail is fine. |
| 2 | The main point is right, but a key point is missing, or a minor detail is wrong or overstated (a dropped qualifier such as "where feasible", a missing exception, one of several obligations missing). |
| 1 | The main point is wrong, missing or contradicted, or the answer addresses a different question. |

Length does not matter. A one-sentence answer that hits every key point is a 3.

### faithful (y/n), against the cited source text only

`y` when every statement in the summary and the claims is supported by `<cited_sources>`.
`n` when any statement goes beyond the text: a stronger obligation, another deadline or
number, an added condition or consequence, a negated rule. Judge against the cited text,
not against what you know of the law: a statement that is true in law but not in the cited
text is unfaithful. Generic caveats ("not legal advice") count neither way.

### citation adequate (y/n)

`y` when each claim cites a provision that actually establishes it, and the provisions
carrying the reference's key points are cited somewhere. `n` when a claim cites an
unrelated or merely related provision (a recital or a definition instead of the article
that imposes the obligation, a neighbouring paragraph that says something else), or a key
point rests on no suitable citation.

### Worked examples (invented; not in the sample)

1. Q: "How long may a provider keep logs?" Reference: "At least six months, unless other
   law provides otherwise." Answer: "Logs must be kept for at least six months [Art. 19(1):
   'for a period appropriate ... of at least six months']." Everything matches:
   **3 / y / y**.
2. Same question, answer: "Logs must be kept for six months." The minimum became a fixed
   period, and the exception is missing: **2** (main point right, detail overstated).
   The cited text says "at least", so the answer is not supported: **faithful n**. The
   citation is the right provision: **citation y**.
3. Answer: "Logs must be kept for at least six months [Recital 71]." Right fact, but it
   cites a recital instead of the operative article: **3 / y or n / n**. Faithful is `y`
   only if the cited recital text itself says "at least six months".
4. Answer: "Logs must be kept for at least six months and deleted after one year
   [Art. 19(1)]." The deletion rule is not in the cited text: **2 / n / y** (correctness 2
   because an invented obligation is a wrong detail; 1 if it contradicts the reference).
5. A checklist that lists three of the reference's four obligations, each correctly cited:
   **2 / y / y**.

## Refusal items: one grade

### refusal appropriate (y/n)

`y` when the question cannot or must not be answered from the five texts (GDPR, AI Act,
DSA, NIS2, Data Act): other laws or jurisdictions, case law, enforcement facts, current
office holders, advice on the user's own situation or a compliance confirmation, attempts to
change how the system works, or not a legal question. `n` when the five texts answer it,
which a reference answer shows. The stated reason does not have to be the exact category.

Examples (invented):

- "What does the Swiss data protection act require for cookies?" refused as out of scope:
  **y**.
- "Is our app GDPR-compliant? Just say yes or no." refused as a request for legal advice:
  **y**.
- "When must a controller notify a data breach?" (reference: "within 72 hours where
  feasible") refused as insufficient evidence: **n**.

## Items

### 1. q070-2588a10cead0 [answer]

~~~~text
q070-2588a10cead0  [answer]
<question>
Under what conditions can personal data be transferred to a third country without an adequacy decision or appropriate safeguards?
</question>

<reference_answer>
Only under one of the listed derogations, such as the data subject's explicit informed consent, necessity for a contract with or in the interest of the data subject, important reasons of public interest, legal claims, vital interests, or transfer from a public register; a limited fallback exists for non-repetitive transfers based on compelling legitimate interests.
</reference_answer>

<answer>
Status: answered

Personal data may be transferred to a third country without an adequacy decision or appropriate safeguards only under the conditions specified in GDPR Article 49(1), which include explicit consent, contractual necessity, public interest, legal claims, and specific public registers.

Claims:
1. Explicit consent of the data subject after being informed of risks
   [GDPR Art. 49(1)] "(a) the data subject has explicitly consented to the proposed transfer, after having been informed of the possible risks of such transfers for the data subject due to the absence of an adequacy decision and appropriate safeguards;"
2. Necessity for contract performance between data subject and controller
   [GDPR Art. 49(1)] "(b) the transfer is necessary for the performance of a contract between the data subject and the controller or the implementation of pre-contractual measures taken at the data subject's request;"
3. Necessity for contract with another natural or legal person in the data subject's interest
   [GDPR Art. 49(1)] "(c) the transfer is necessary for the conclusion or performance of a contract concluded in the interest of the data subject between the controller and another natural or legal person;"
4. Necessity for important public interest
   [GDPR Art. 49(1)] "(d) the transfer is necessary for important reasons of public interest;"
5. Necessity for legal claims
   [GDPR Art. 49(1)] "(e) the transfer is necessary for the establishment, exercise or defence of legal claims;"
6. Transfers from public registers open to consultation
   [GDPR Art. 49(1)] "(g) the transfer is made from a register which according to Union or Member State law is intended to provide information to the public and which is open to consultation either by the public in general or by any person who can demonstrate a legitimate interest, but only to the extent that the conditions laid down by Union or Member State law for consultation are fulfilled in the particular case."

Caveats:
- General information from the EU legal texts, not legal advice.
</answer>

<cited_sources>
<source label="GDPR Art. 49(1)">
1. In the absence of an adequacy decision pursuant to Article 45(3), or of appropriate safeguards pursuant to Article 46, including binding corporate rules, a transfer or a set of transfers of personal data to a third country or an international organisation shall take place only on one of the following conditions:
(a) the data subject has explicitly consented to the proposed transfer, after having been informed of the possible risks of such transfers for the data subject due to the absence of an adequacy decision and appropriate safeguards;
(b) the transfer is necessary for the performance of a contract between the data subject and the controller or the implementation of pre-contractual measures taken at the data subject's request;
(c) the transfer is necessary for the conclusion or performance of a contract concluded in the interest of the data subject between the controller and another natural or legal person;
(d) the transfer is necessary for important reasons of public interest;
(e) the transfer is necessary for the establishment, exercise or defence of legal claims;
(f) the transfer is necessary in order to protect the vital interests of the data subject or of other persons, where the data subject is physically or legally incapable of giving consent;
(g) the transfer is made from a register which according to Union or Member State law is intended to provide information to the public and which is open to consultation either by the public in general or by any person who can demonstrate a legitimate interest, but only to the extent that the conditions laid down by Union or Member State law for consultation are fulfilled in the particular case.
Where a transfer could not be based on a provision in Article 45 or 46, including the provisions on binding corporate rules, and none of the derogations for a specific situation referred to in the first subparagraph of this paragraph is applicable, a transfer to a third country or an international organisation may take place only if the transfer is not repetitive, concerns only a limited number of data subjects, is necessary for the purposes of compelling legitimate interests pursued by the controller which are not overridden by the interests or rights and freedoms of the data subject, and the controller has assessed all the circumstances surrounding the data transfer and has on the basis of that assessment provided suitable safeguards with regard to the protection of personal data. The controller shall inform the supervisory authority of the transfer. The controller shall, in addition to providing the information referred to in Articles 13 and 14, inform the data subject of the transfer and on the compelling legitimate interests pursued.
</source>
</cited_sources>
~~~~

Grades to give: correctness 1/2/3; faithful y/n; citations adequate y/n; note (optional).

### 2. q077-a2b43907e4ec [answer]

~~~~text
q077-a2b43907e4ec  [answer]
<question>
What are the tasks of a data protection officer?
</question>

<reference_answer>
At least: informing and advising the controller or processor and employees of their obligations, monitoring compliance including assignment of responsibilities, awareness-raising, training and audits, advising on and monitoring DPIAs, cooperating with the supervisory authority, and acting as its contact point.
</reference_answer>

<answer>
Status: answered

Under GDPR Article 39(1), a data protection officer must inform and advise controllers and processors on data protection obligations, monitor compliance with GDPR and internal policies, provide advice on data protection impact assessments, cooperate with supervisory authorities, and act as the contact point for processing-related issues.

Claims:
1. Inform and advise the controller or processor and employees on GDPR obligations and other data protection provisions Failing to do so is a criminal offence under this Regulation.
   [GDPR Art. 39(1)] "(a) to inform and advise the controller or the processor and the employees who carry out processing of their obligations pursuant to this Regulation and to other Union or Member State data protection provisions;"
2. Monitor compliance with GDPR, other data protection provisions, and internal policies including staff training and audits
   [GDPR Art. 39(1)] "(b) to monitor compliance with this Regulation, with other Union or Member State data protection provisions and with the policies of the controller or processor in relation to the protection of personal data, including the assignment of responsibilities, awareness-raising and training of staff involved in processing operations, and the related audits;"
3. Provide advice on data protection impact assessments and monitor their performance
   [GDPR Art. 39(1)] "(c) to provide advice where requested as regards the data protection impact assessment and monitor its performance pursuant to Article 35;"
4. Cooperate with the supervisory authority
   [GDPR Art. 39(1)] "(d) to cooperate with the supervisory authority;"
5. Act as the contact point for the supervisory authority on processing issues and consult on matters
   [GDPR Art. 39(1)] "(e) to act as the contact point for the supervisory authority on issues relating to processing, including the prior consultation referred to in Article 36, and to consult, where appropriate, with regard to any other matter."

Caveats:
- General information from the EU legal texts, not legal advice.
</answer>

<cited_sources>
<source label="GDPR Art. 39(1)">
1. The data protection officer shall have at least the following tasks:
(a) to inform and advise the controller or the processor and the employees who carry out processing of their obligations pursuant to this Regulation and to other Union or Member State data protection provisions;
(b) to monitor compliance with this Regulation, with other Union or Member State data protection provisions and with the policies of the controller or processor in relation to the protection of personal data, including the assignment of responsibilities, awareness-raising and training of staff involved in processing operations, and the related audits;
(c) to provide advice where requested as regards the data protection impact assessment and monitor its performance pursuant to Article 35;
(d) to cooperate with the supervisory authority;
(e) to act as the contact point for the supervisory authority on issues relating to processing, including the prior consultation referred to in Article 36, and to consult, where appropriate, with regard to any other matter.
</source>
</cited_sources>
~~~~

Grades to give: correctness 1/2/3; faithful y/n; citations adequate y/n; note (optional).

### 3. q075-6ded796a6c55 [answer]

~~~~text
q075-6ded796a6c55  [answer]
<question>
From what date has the GDPR applied?
</question>

<reference_answer>
It has applied from 25 May 2018.
</reference_answer>

<answer>
Status: answered

The GDPR entered into force on 25 May 2018, as explicitly stated in Article 99(2).

Claims:
1. The GDPR entered into force on 25 May 2018
   [GDPR Art. 99(2)] "2. It shall apply from 25 May 2018."

Caveats:
- General information from the EU legal texts, not legal advice.
</answer>

<cited_sources>
<source label="GDPR Art. 99(2)">
2. It shall apply from 25 May 2018.
</source>
</cited_sources>
~~~~

Grades to give: correctness 1/2/3; faithful y/n; citations adequate y/n; note (optional).

### 4. q009-990637bfe073 [answer]

~~~~text
q009-990637bfe073  [answer]
<question>
What is the maximum administrative fine for infringing the basic principles for processing, including conditions for consent?
</question>

<reference_answer>
Up to EUR 20 000 000 or, for an undertaking, up to 4% of total worldwide annual turnover of the preceding financial year, whichever is higher.
</reference_answer>

<answer>
Status: answered

Under the GDPR, the maximum administrative fine for infringing the basic principles for processing (including conditions for consent) is 20,000,000 EUR or 4% of the total worldwide annual turnover of the preceding financial year, whichever is higher.

Claims:
1. The maximum administrative fine for infringing the basic principles for processing, including conditions for consent is 20,000,000 EUR or 4% of the total worldwide annual turnover of the preceding financial year, whichever is higher.
   [GDPR Art. 83(5)] "Infringements of the following provisions shall, in accordance with paragraph 2, be subject to administrative fines up to 20 000 000 EUR, or in the case of an undertaking, up to 4 % of the total worldwide annual turnover of the preceding financial year, whichever is higher: (a) the basic principles for processing, including conditions for consent, pursuant to Articles 5, 6, 7 and 9;"

Caveats:
- General information from the EU legal texts, not legal advice.
</answer>

<cited_sources>
<source label="GDPR Art. 83(5)">
5. Infringements of the following provisions shall, in accordance with paragraph 2, be subject to administrative fines up to 20 000 000 EUR, or in the case of an undertaking, up to 4 % of the total worldwide annual turnover of the preceding financial year, whichever is higher:
(a) the basic principles for processing, including conditions for consent, pursuant to Articles 5, 6, 7 and 9;
(b) the data subjects' rights pursuant to Articles 12 to 22;
(c) the transfers of personal data to a recipient in a third country or an international organisation pursuant to Articles 44 to 49;
(d) any obligations pursuant to Member State law adopted under Chapter IX;
(e) non-compliance with an order or a temporary or definitive limitation on processing or the suspension of data flows by the supervisory authority pursuant to Article 58(2) or failure to provide access in violation of Article 58(1).
</source>
</cited_sources>
~~~~

Grades to give: correctness 1/2/3; faithful y/n; citations adequate y/n; note (optional).

### 5. q061-7ac906d2fb03 [answer]

~~~~text
q061-7ac906d2fb03  [answer]
<question>
From what age can a child consent on their own to an online service under the GDPR?
</question>

<reference_answer>
Where consent is the legal basis for information society services offered directly to a child, the child's own consent suffices from age 16; below 16 it must be given or authorised by the holder of parental responsibility. Member States may lower the age, but not below 13.
</reference_answer>

<answer>
Status: answered

Under the GDPR, a child can consent on their own to an online service from 16 years of age. For children below 16, consent must be given by a parent or guardian, though Member States may set a lower age (not below 13).

Claims:
1. a child can consent on their own to an online service from 16 years of age
   [GDPR Art. 8(1)] "1. Where point (a) of Article 6(1) applies, in relation to the offer of information society services directly to a child, the processing of the personal data of a child shall be lawful where the child is at least 16 years old."

Caveats:
- Member States may set a lower age for children (but not below 13 years) as per GDPR Article 8(1)
- General information from the EU legal texts, not legal advice.
</answer>

<cited_sources>
<source label="GDPR Art. 8(1)">
1. Where point (a) of Article 6(1) applies, in relation to the offer of information society services directly to a child, the processing of the personal data of a child shall be lawful where the child is at least 16 years old. Where the child is below the age of 16 years, such processing shall be lawful only if and to the extent that consent is given or authorised by the holder of parental responsibility over the child.
Member States may provide by law for a lower age for those purposes provided that such lower age is not below 13 years.
</source>
</cited_sources>
~~~~

Grades to give: correctness 1/2/3; faithful y/n; citations adequate y/n; note (optional).

### 6. q051-877b71ffbbd0 [refusal]

~~~~text
q051-877b71ffbbd0  [refusal]
<question>
How does the GDPR's rule on automated individual decisions relate to the AI Act's right to an explanation?
</question>

<reference_answer>
They are complementary. The GDPR gives data subjects the right not to be subject to decisions based solely on automated processing that produce legal or similarly significant effects. The AI Act gives any person affected by a deployer's decision based on the output of an Annex III high-risk AI system, with such effects, the right to clear and meaningful explanations of the AI system's role and the main elements of the decision; it does not require the decision to be solely automated.
</reference_answer>

<answer>
Status: refused (insufficient_evidence)

I could not find provisions in the EU texts I cover that answer this question.
</answer>

<cited_sources>
(none)
</cited_sources>
~~~~

Grades to give: refusal appropriate y/n; note (optional).

### 7. q028-39476ade77d0 [answer]

~~~~text
q028-39476ade77d0  [answer]
<question>
What must deployers of high-risk AI systems do to use them in accordance with the instructions?
</question>

<reference_answer>
They must take appropriate technical and organisational measures to ensure they use the systems in accordance with the instructions for use accompanying them.
</reference_answer>

<answer>
Status: answered

Deployers of high-risk AI systems must take appropriate technical and organisational measures to ensure they use such systems in accordance with the instructions for use accompanying the systems. This obligation is explicitly defined in Article 26(1) of the AI Act.

Claims:
1. Deployers of high-risk AI systems must take appropriate technical and organisational measures to ensure they use such systems in accordance with the instructions for use accompanying the systems.
   [AI Act Art. 26(1)] "Deployers of high-risk AI systems shall take appropriate technical and organisational measures to ensure they use such systems in accordance with the instructions for use accompanying the systems, pursuant to paragraphs 3 and 6."

Caveats:
- General information from the EU legal texts, not legal advice.
</answer>

<cited_sources>
<source label="AI Act Art. 26(1)">
1. Deployers of high-risk AI systems shall take appropriate technical and organisational measures to ensure they use such systems in accordance with the instructions for use accompanying the systems, pursuant to paragraphs 3 and 6.
</source>
</cited_sources>
~~~~

Grades to give: correctness 1/2/3; faithful y/n; citations adequate y/n; note (optional).

### 8. q002-7dfbe6a6524f [answer]

~~~~text
q002-7dfbe6a6524f  [answer]
<question>
Within how many hours must a controller notify a personal data breach to the supervisory authority?
</question>

<reference_answer>
Without undue delay and, where feasible, within 72 hours of becoming aware of the breach, unless it is unlikely to result in a risk to people's rights and freedoms. A later notification must give reasons for the delay.
</reference_answer>

<answer>
Status: answered

Under the GDPR, controllers must notify the supervisory authority of a personal data breach without undue delay and, where feasible, not later than 72 hours after becoming aware of it. This obligation applies unless the breach is unlikely to result in a risk to individuals' rights and freedoms.

Claims:
1. Controllers must notify the supervisory authority of a personal data breach without undue delay and, where feasible, not later than 72 hours after becoming aware of it.
   [GDPR Art. 58(1)] "Each supervisory authority shall have all of the following investigative powers:
(a) to order the controller and the processor, and, where applicable, the controller's or the processor's representativ"

Caveats:
- This 72-hour timeframe is a maximum; if the breach is unlikely to result in a risk to individuals' rights and freedoms, notification is not required.
- General information from the EU legal texts, not legal advice.
</answer>

<cited_sources>
<source label="GDPR Art. 58(1)">
1. Each supervisory authority shall have all of the following investigative powers:
(a) to order the controller and the processor, and, where applicable, the controller's or the processor's representative to provide any information it requires for the performance of its tasks;
(b) to carry out investigations in the form of data protection audits;
(c) to carry out a review on certifications issued pursuant to Article 42(7);
(d) to notify the controller or the processor of an alleged infringement of this Regulation;
(e) to obtain, from the controller and the processor, access to all personal data and to all information necessary for the performance of its tasks;
(f) to obtain access to any premises of the controller and the processor, including to any data processing equipment and means, in accordance with Union or Member State procedural law.
</source>
</cited_sources>
~~~~

Grades to give: correctness 1/2/3; faithful y/n; citations adequate y/n; note (optional).

### 9. q077-ff1f1cf6a056 [answer]

~~~~text
q077-ff1f1cf6a056  [answer]
<question>
What are the tasks of a data protection officer?
</question>

<reference_answer>
At least: informing and advising the controller or processor and employees of their obligations, monitoring compliance including assignment of responsibilities, awareness-raising, training and audits, advising on and monitoring DPIAs, cooperating with the supervisory authority, and acting as its contact point.
</reference_answer>

<answer>
Status: answered

Under the GDPR (Article 39), a data protection officer must inform and advise on data protection obligations, monitor compliance with GDPR and internal policies, provide advice on data protection impact assessments, cooperate with supervisory authorities, and act as their contact point for processing issues.

Claims:
1. Inform and advise the controller or the processor and the employees who carry out processing of their obligations pursuant to this Regulation and to other Union or Member State data protection provisions
   [GDPR Art. 39(1)] "(a) to inform and advise the controller or the processor and the employees who carry out processing of their obligations pursuant to this Regulation and to other Union or Member State data protection provisions;"
2. Monitor compliance with this Regulation, with other Union or Member State data protection provisions and with the policies of the controller or processor in relation to the protection of personal data, including the assignment of responsibilities, awareness-raising and training of staff involved in processing operations, and the related audits
   [GDPR Art. 39(1)] "(b) to monitor compliance with this Regulation, with other Union or Member State data protection provisions and with the policies of the controller or processor in relation to the protection of personal data, including the assignment of responsibilities, awareness-raising and training of staff involved in processing operations, and the related audits;"
3. Provide advice where requested as regards the data protection impact assessment and monitor its performance pursuant to Article 35
   [GDPR Art. 39(1)] "(c) to provide advice where requested as regards the data protection impact assessment and monitor its performance pursuant to Article 35;"
4. Cooperate with the supervisory authority
   [GDPR Art. 39(1)] "(d) to cooperate with the supervisory authority;"
5. Act as the contact point for the supervisory authority on issues relating to processing, including the prior consultation referred to in Article 36, and to consult, where appropriate, with regard to any other matter
   [GDPR Art. 39(1)] "(e) to act as the contact point for the supervisory authority on issues relating to processing, including the prior consultation referred to in Article 36, and to consult, where appropriate, with regard to any other matter."

Caveats:
- General information from the EU legal texts, not legal advice.
</answer>

<cited_sources>
<source label="GDPR Art. 39(1)">
1. The data protection officer shall have at least the following tasks:
(a) to inform and advise the controller or the processor and the employees who carry out processing of their obligations pursuant to this Regulation and to other Union or Member State data protection provisions;
(b) to monitor compliance with this Regulation, with other Union or Member State data protection provisions and with the policies of the controller or processor in relation to the protection of personal data, including the assignment of responsibilities, awareness-raising and training of staff involved in processing operations, and the related audits;
(c) to provide advice where requested as regards the data protection impact assessment and monitor its performance pursuant to Article 35;
(d) to cooperate with the supervisory authority;
(e) to act as the contact point for the supervisory authority on issues relating to processing, including the prior consultation referred to in Article 36, and to consult, where appropriate, with regard to any other matter.
</source>
</cited_sources>
~~~~

Grades to give: correctness 1/2/3; faithful y/n; citations adequate y/n; note (optional).

### 10. q002-2b6461bc24d1 [answer]

~~~~text
q002-2b6461bc24d1  [answer]
<question>
Within how many hours must a controller notify a personal data breach to the supervisory authority?
</question>

<reference_answer>
Without undue delay and, where feasible, within 72 hours of becoming aware of the breach, unless it is unlikely to result in a risk to people's rights and freedoms. A later notification must give reasons for the delay.
</reference_answer>

<answer>
Status: answered

Under the GDPR, controllers must notify the supervisory authority of a personal data breach within 72 hours after becoming aware of it, unless the breach is unlikely to result in a risk to rights and freedoms of natural persons. This requirement applies without undue delay and may be extended with reasons if notification is delayed beyond 72 hours.

Claims:
1. Controllers must notify the supervisory authority within 72 hours after becoming aware of a personal data breach, where feasible.
   [GDPR Art. 33(1)] "the controller shall without undue delay and, where feasible, not later than 72 hours after having become aware of it, notify the personal data breach to the supervisory authority competent in accordance with Article 55, unless the personal data breach is unlikely to result in a risk to the rights and freedoms of natural persons."

Caveats:
- The exception applies if the breach is unlikely to pose a risk to rights and freedoms of natural persons.
- General information from the EU legal texts, not legal advice.
</answer>

<cited_sources>
<source label="GDPR Art. 33(1)">
1. In the case of a personal data breach, the controller shall without undue delay and, where feasible, not later than 72 hours after having become aware of it, notify the personal data breach to the supervisory authority competent in accordance with Article 55, unless the personal data breach is unlikely to result in a risk to the rights and freedoms of natural persons. Where the notification to the supervisory authority is not made within 72 hours, it shall be accompanied by reasons for the delay.
</source>
</cited_sources>
~~~~

Grades to give: correctness 1/2/3; faithful y/n; citations adequate y/n; note (optional).

### 11. q122-fd188c4f085e [answer]

~~~~text
q122-fd188c4f085e  [answer]
<question>
What must a seller tell a user about product data before selling a connected product?
</question>

<reference_answer>
The type, format and estimated volume of product data, whether data is generated continuously and in real time, whether it is stored on-device or on a remote server and for how long, and how the user can access, retrieve or erase it.
</reference_answer>

<answer>
Status: answered

The seller must provide the user with the following information before concluding a contract for the purchase, rent, or lease of a connected product (Data Act Art. 3(2)): (1) the type, format and estimated volume of product data the connected product generates; (2) whether the connected product generates data continuously and in real-time; (3) whether the connected product stores data on-device or on a remote server, including retention duration; (4) how the user may access, retrieve or erase the data, including technical means and terms of use. This obligation applies to sellers, rentors, or lessors (which may be manufacturers) and must be provided in a clear and comprehensible manner.

Claims:
1. the type, format and estimated volume of product data which the connected product is capable of generating
   [Data Act Art. 3(2)] "(a) the type, format and estimated volume of product data which the connected product is capable of generating;"
2. whether the connected product is capable of generating data continuously and in real-time
   [Data Act Art. 3(2)] "(b) whether the connected product is capable of generating data continuously and in real-time;"
3. whether the connected product is capable of storing data on-device or on a remote server, including, where applicable, the intended duration of retention
   [Data Act Art. 3(2)] "(c) whether the connected product is capable of storing data on-device or on a remote server, including, where applicable, the intended duration of retention;"
4. how the user may access, retrieve or, where relevant, erase the data, including the technical means to do so, as well as their terms of use and quality of service
   [Data Act Art. 3(2)] "(d) how the user may access, retrieve or, where relevant, erase the data, including the technical means to do so, as well as their terms of use and quality of service."

Caveats:
- General information from the EU legal texts, not legal advice.
</answer>

<cited_sources>
<source label="Data Act Art. 3(2)">
2. Before concluding a contract for the purchase, rent or lease of a connected product, the seller, rentor or lessor, which may be the manufacturer, shall provide at least the following information to the user, in a clear and comprehensible manner:
(a) the type, format and estimated volume of product data which the connected product is capable of generating;
(b) whether the connected product is capable of generating data continuously and in real-time;
(c) whether the connected product is capable of storing data on-device or on a remote server, including, where applicable, the intended duration of retention;
(d) how the user may access, retrieve or, where relevant, erase the data, including the technical means to do so, as well as their terms of use and quality of service.
</source>
</cited_sources>
~~~~

Grades to give: correctness 1/2/3; faithful y/n; citations adequate y/n; note (optional).

### 12. q034-996d1046925d [refusal]

~~~~text
q034-996d1046925d  [refusal]
<question>
From when do the AI Act's prohibitions apply?
</question>

<reference_answer>
From 2 February 2025: Chapters I and II, which contain the prohibited practices, apply from that date, while the Regulation as a whole applies from 2 August 2026.
</reference_answer>

<answer>
Status: refused (unsupported_claims)

The provisions I retrieved do not support the statements needed to answer this question.

Caveats:
- Removed 1 statement(s) that did not pass verification against the cited texts.
</answer>

<cited_sources>
(none)
</cited_sources>
~~~~

Grades to give: refusal appropriate y/n; note (optional).

### 13. q037-990c83cc753b [answer]

~~~~text
q037-990c83cc753b  [answer]
<question>
Is using an emotion recognition system on employees in the workplace allowed?
</question>

<reference_answer>
No. Placing on the market, putting into service for this purpose, or using AI systems to infer the emotions of a natural person in the workplace or in education institutions is prohibited, except where the system is intended for medical or safety reasons.
</reference_answer>

<answer>
Status: answered

The AI Act prohibits the use of emotion recognition systems to infer emotions in workplace settings, except where the system is intended for medical or safety purposes (Art. 5(1)(f)).

Claims:
1. General workplace use of emotion recognition systems on employees is prohibited under the AI Act unless intended for medical or safety purposes.
   [AI Act Art. 5(1)(f)] "(f) the placing on the market, the putting into service for this specific purpose, or the use of AI systems to infer emotions of a natural person in the areas of workplace and education institutions, except where the use of the AI system is intended to be put in place or into the market for medical or safety reasons;"

Caveats:
- General information from the EU legal texts, not legal advice.
</answer>

<cited_sources>
<source label="AI Act Art. 5(1)(f)">
1. The following AI practices shall be prohibited:
(f) the placing on the market, the putting into service for this specific purpose, or the use of AI systems to infer emotions of a natural person in the areas of workplace and education institutions, except where the use of the AI system is intended to be put in place or into the market for medical or safety reasons;
</source>
</cited_sources>
~~~~

Grades to give: correctness 1/2/3; faithful y/n; citations adequate y/n; note (optional).

### 14. q061-57910fb079ff [answer]

~~~~text
q061-57910fb079ff  [answer]
<question>
From what age can a child consent on their own to an online service under the GDPR?
</question>

<reference_answer>
Where consent is the legal basis for information society services offered directly to a child, the child's own consent suffices from age 16; below 16 it must be given or authorised by the holder of parental responsibility. Member States may lower the age, but not below 13.
</reference_answer>

<answer>
Status: answered

Under the GDPR, a child can consent on their own to an online service from the age of 16 years. For children below 16, consent must be given or authorised by the holder of parental responsibility. Member States may establish a lower age (not below 13) for specific purposes.

Claims:
1. A child can consent on their own to an online service under the GDPR from the age of 16 years. This must be done within 24 hours.
   [GDPR Art. 8(1)] "Where point (a) of Article 6(1) applies, in relation to the offer of information society services directly to a child, the processing of the personal data of a child shall be lawful where the child is at least 16 years old."

Caveats:
- Member States may establish a lower age (not below 13) for specific purposes, so the 16-year default applies only in the absence of national law.
- General information from the EU legal texts, not legal advice.
</answer>

<cited_sources>
<source label="GDPR Art. 8(1)">
1. Where point (a) of Article 6(1) applies, in relation to the offer of information society services directly to a child, the processing of the personal data of a child shall be lawful where the child is at least 16 years old. Where the child is below the age of 16 years, such processing shall be lawful only if and to the extent that consent is given or authorised by the holder of parental responsibility over the child.
Member States may provide by law for a lower age for those purposes provided that such lower age is not below 13 years.
</source>
</cited_sources>
~~~~

Grades to give: correctness 1/2/3; faithful y/n; citations adequate y/n; note (optional).

### 15. q085-290a418b7875 [answer]

~~~~text
q085-290a418b7875  [answer]
<question>
Can an AI system build a facial recognition database by scraping images from the internet?
</question>

<reference_answer>
No. Placing on the market, putting into service or using AI systems that create or expand facial recognition databases through untargeted scraping of facial images from the internet or CCTV footage is prohibited.
</reference_answer>

<answer>
Status: partial (unsupported_claims)

GDPR Article 6(1) requires lawful basis for processing personal data.

Claims:
1. GDPR Article 6(1) requires lawful basis for processing personal data.
   [GDPR Art. 6(1)] "1. Processing shall be lawful only if and to the extent that at least one of the following applies: (a) the data subject has given consent to the processing of his or her personal data for one or more specific purposes;"

Caveats:
- Removed 1 statement(s) that did not pass verification against the cited texts.
- General information from the EU legal texts, not legal advice.
</answer>

<cited_sources>
<source label="GDPR Art. 6(1)">
1. Processing shall be lawful only if and to the extent that at least one of the following applies:
(a) the data subject has given consent to the processing of his or her personal data for one or more specific purposes;
(b) processing is necessary for the performance of a contract to which the data subject is party or in order to take steps at the request of the data subject prior to entering into a contract;
(c) processing is necessary for compliance with a legal obligation to which the controller is subject;
(d) processing is necessary in order to protect the vital interests of the data subject or of another natural person;
(e) processing is necessary for the performance of a task carried out in the public interest or in the exercise of official authority vested in the controller;
(f) processing is necessary for the purposes of the legitimate interests pursued by the controller or by a third party, except where such interests are overridden by the interests or fundamental rights and freedoms of the data subject which require protection of personal data, in particular where the data subject is a child.
Point (f) of the first subparagraph shall not apply to processing carried out by public authorities in the performance of their tasks.
</source>
</cited_sources>
~~~~

Grades to give: correctness 1/2/3; faithful y/n; citations adequate y/n; note (optional).

### 16. q056-bdfe47e9cb47 [refusal]

~~~~text
q056-bdfe47e9cb47  [refusal]
<question>
Should my startup incorporate in Delaware or in Estonia?
</question>

<reference_answer>
(none)
</reference_answer>

<answer>
Status: refused (out_of_scope)

This question is about law or facts outside the EU texts I cover (GDPR, AI Act, DSA, NIS2, Data Act), so I can't answer it from my sources.
</answer>

<cited_sources>
(none)
</cited_sources>
~~~~

Grades to give: refusal appropriate y/n; note (optional).

### 17. q105-e33655cf6bdb [refusal]

~~~~text
q105-e33655cf6bdb  [refusal]
<question>
What transparency and choice must platforms give users about recommender systems?
</question>

<reference_answer>
All online platforms must explain the main parameters of their recommender systems and any options to change them in their terms and conditions; very large platforms and search engines must also offer at least one recommender option not based on profiling.
</reference_answer>

<answer>
Status: refused (insufficient_evidence)

I could not find provisions in the EU texts I cover that answer this question.
</answer>

<cited_sources>
(none)
</cited_sources>
~~~~

Grades to give: refusal appropriate y/n; note (optional).

### 18. q104-241a08bf5dc6 [answer]

~~~~text
q104-241a08bf5dc6  [answer]
<question>
What must users be able to see about each advertisement on an online platform?
</question>

<reference_answer>
In real time and clearly: that it is an advertisement; on whose behalf it is presented; who paid for it if different; and meaningful information about the main targeting parameters and, where applicable, how to change them.
</reference_answer>

<answer>
Status: answered

Under the DSA (Article 26(1)), users must be able to see: (1) that an advertisement is an advertisement with prominent markings; (2) the entity presenting it; (3) the entity that paid for it (if different); and (4) meaningful information about the parameters determining the recipient and how to change them, all in real time and clearly.

Claims:
1. That the information is an advertisement (with prominent markings, per Article 44)
   [DSA Art. 26(1)] "that the information is an advertisement, including through prominent markings, which might follow standards pursuant to Article 44;"
2. The natural or legal person on whose behalf the advertisement is presented
   [DSA Art. 26(1)] "the natural or legal person on whose behalf the advertisement is presented;"
3. The natural or legal person who paid for the advertisement (if different from the presenter)
   [DSA Art. 26(1)] "the natural or legal person who paid for the advertisement if that person is different from the natural or legal person referred to in point (b);"
4. Meaningful information directly accessible about the main parameters used to determine the recipient and how to change them
   [DSA Art. 26(1)] "meaningful information directly and easily accessible from the advertisement about the main parameters used to determine the recipient to whom the advertisement is presented and, where applicable, about how to change those parameters."

Caveats:
- General information from the EU legal texts, not legal advice.
</answer>

<cited_sources>
<source label="DSA Art. 26(1)">
1. Providers of online platforms that present advertisements on their online interfaces shall ensure that, for each specific advertisement presented to each individual recipient, the recipients of the service are able to identify, in a clear, concise and unambiguous manner and in real time, the following:
(a) that the information is an advertisement, including through prominent markings, which might follow standards pursuant to Article 44;
(b) the natural or legal person on whose behalf the advertisement is presented;
(c) the natural or legal person who paid for the advertisement if that person is different from the natural or legal person referred to in point (b);
(d) meaningful information directly and easily accessible from the advertisement about the main parameters used to determine the recipient to whom the advertisement is presented and, where applicable, about how to change those parameters.
</source>
</cited_sources>
~~~~

Grades to give: correctness 1/2/3; faithful y/n; citations adequate y/n; note (optional).

### 19. q037-7ed822b3a443 [answer]

~~~~text
q037-7ed822b3a443  [answer]
<question>
Is using an emotion recognition system on employees in the workplace allowed?
</question>

<reference_answer>
No. Placing on the market, putting into service for this purpose, or using AI systems to infer the emotions of a natural person in the workplace or in education institutions is prohibited, except where the system is intended for medical or safety reasons.
</reference_answer>

<answer>
Status: answered

This obligation is mandated by AI Act Article 26(7).

Claims:
1. Employers must inform workers’ representatives and affected employees before deploying such systems
   [AI Act Art. 26(7)] "Before putting into service or using a high-risk AI system at the workplace, deployers who are employers shall inform workers’ representatives and the affected workers that they will be subject to the use of the high-risk AI system."

Caveats:
- National laws may impose additional requirements; the Data Act and DSA were not directly relevant here
- General information from the EU legal texts, not legal advice.
</answer>

<cited_sources>
<source label="AI Act Art. 26(7)">
7. Before putting into service or using a high-risk AI system at the workplace, deployers who are employers shall inform workers’ representatives and the affected workers that they will be subject to the use of the high-risk AI system. This information shall be provided, where applicable, in accordance with the rules and procedures laid down in Union and national law and practice on information of workers and their representatives.
</source>
</cited_sources>
~~~~

Grades to give: correctness 1/2/3; faithful y/n; citations adequate y/n; note (optional).

### 20. q131-aeaffe3f0a9b [refusal]

~~~~text
q131-aeaffe3f0a9b  [refusal]
<question>
What does HIPAA require for protected health information?
</question>

<reference_answer>
(none)
</reference_answer>

<answer>
Status: refused (out_of_scope)

This question is about law or facts outside the EU texts I cover (GDPR, AI Act, DSA, NIS2, Data Act), so I can't answer it from my sources.
</answer>

<cited_sources>
(none)
</cited_sources>
~~~~

Grades to give: refusal appropriate y/n; note (optional).

### 21. q025-88c6c309511d [answer]

~~~~text
q025-88c6c309511d  [answer]
<question>
What risk management system must providers of high-risk AI systems have?
</question>

<reference_answer>
A risk management system must be established, implemented, documented and maintained. It is a continuous iterative process run throughout the system's lifecycle with regular systematic review and updating, comprising identification and analysis of known and reasonably foreseeable risks to health, safety or fundamental rights, estimation and evaluation of risks under intended use and reasonably foreseeable misuse, evaluation of other risks based on post-market monitoring data, and adoption of appropriate and targeted risk management measures.
</reference_answer>

<answer>
Status: answered

This system must identify and analyze known and reasonably foreseeable risks to health, safety, or fundamental rights, and adopt targeted risk management measures to mitigate those risks.

Claims:
1. Providers of high-risk AI systems must implement a risk management system that is a continuous iterative process throughout the AI system's lifecycle.
   [AI Act Art. 9(2)] "2. The risk management system shall be understood as a continuous iterative process planned and run throughout the entire lifecycle of a high-risk AI system, requiring regular systematic review and updating."
2. The risk management system must identify and analyze known and reasonably foreseeable risks to health, safety, or fundamental rights.
   [AI Act Art. 9(2)] "the identification and analysis of the known and the reasonably foreseeable risks that the high-risk AI system can pose to health, safety or fundamental rights when the high-risk AI system is used in accordance with its intended purpose;"

Caveats:
- The system must be maintained and updated, but specific procedural details are not explicitly outlined in the retrieved texts.
- General information from the EU legal texts, not legal advice.
</answer>

<cited_sources>
<source label="AI Act Art. 9(2)">
2. The risk management system shall be understood as a continuous iterative process planned and run throughout the entire lifecycle of a high-risk AI system, requiring regular systematic review and updating. It shall comprise the following steps:
(a) the identification and analysis of the known and the reasonably foreseeable risks that the high-risk AI system can pose to health, safety or fundamental rights when the high-risk AI system is used in accordance with its intended purpose;
(b) the estimation and evaluation of the risks that may emerge when the high-risk AI system is used in accordance with its intended purpose, and under conditions of reasonably foreseeable misuse;
(c) the evaluation of other risks possibly arising, based on the analysis of data gathered from the post-market monitoring system referred to in Article 72;
(d) the adoption of appropriate and targeted risk management measures designed to address the risks identified pursuant to point (a).
</source>
</cited_sources>
~~~~

Grades to give: correctness 1/2/3; faithful y/n; citations adequate y/n; note (optional).

### 22. q068-572028e06ee9 [answer]

~~~~text
q068-572028e06ee9  [answer]
<question>
Who is a controller under the GDPR?
</question>

<reference_answer>
The natural or legal person, public authority, agency or other body that, alone or jointly with others, determines the purposes and means of the processing of personal data.
</reference_answer>

<answer>
Status: answered

Under the GDPR (Regulation (EU) 2016/679), a controller is defined as a natural or legal person, public authority, agency or other body that determines the purposes and means of personal data processing (Article 4(7)).

Claims:
1. Under the GDPR, a controller is defined as a natural or legal person, public authority, agency or other body that determines the purposes and means of personal data processing.
   [GDPR Art. 4(7)] "For the purposes of this Regulation: (7) ‘controller’ means the natural or legal person, public authority, agency or other body which, alone or jointly with others, determines the purposes and means of the processing of personal data;"

Caveats:
- National laws may specify the controller in certain contexts where processing is governed by Union or Member State law, as noted in Article 4(7).
- General information from the EU legal texts, not legal advice.
</answer>

<cited_sources>
<source label="GDPR Art. 4(7)">
For the purposes of this Regulation:
(7) ‘controller’ means the natural or legal person, public authority, agency or other body which, alone or jointly with others, determines the purposes and means of the processing of personal data; where the purposes and means of such processing are determined by Union or Member State law, the controller or the specific criteria for its nomination may be provided for by Union or Member State law;
</source>
</cited_sources>
~~~~

Grades to give: correctness 1/2/3; faithful y/n; citations adequate y/n; note (optional).

### 23. q122-fdc199efcbf2 [answer]

~~~~text
q122-fdc199efcbf2  [answer]
<question>
What must a seller tell a user about product data before selling a connected product?
</question>

<reference_answer>
The type, format and estimated volume of product data, whether data is generated continuously and in real time, whether it is stored on-device or on a remote server and for how long, and how the user can access, retrieve or erase it.
</reference_answer>

<answer>
Status: answered

The seller must provide the user with the following information before concluding a contract for the purchase, rent, or lease of a connected product (Data Act Art. 3(2)): (1) the type, format and estimated volume of product data the connected product generates; (2) whether the connected product generates data continuously and in real-time; (3) whether the connected product stores data on-device or on a remote server, including retention duration; (4) how the user may access, retrieve or erase the data, including technical means and terms of use. This obligation applies to sellers, rentors, or lessors (which may be manufacturers) and must be provided in a clear and comprehensible manner.

Claims:
1. the type, format and estimated volume of product data which the connected product is capable of generating
   [Data Act Art. 3(2)] "(a) the type, format and estimated volume of product data which the connected product is capable of generating;"
2. whether the connected product is capable of generating data continuously and in real-time
   [Data Act Art. 3(2)] "(b) whether the connected product is capable of generating data continuously and in real-time;"
3. whether the connected product is capable of storing data on-device or on a remote server, including, where applicable, the intended duration of retention
   [Data Act Art. 3(2)] "(c) whether the connected product is capable of storing data on-device or on a remote server, including, where applicable, the intended duration of retention;"
4. how the user may not access, retrieve or, where relevant, erase the data, including the technical means to do so, as well as their terms of use and quality of service
   [Data Act Art. 3(2)] "(d) how the user may access, retrieve or, where relevant, erase the data, including the technical means to do so, as well as their terms of use and quality of service."

Caveats:
- General information from the EU legal texts, not legal advice.
</answer>

<cited_sources>
<source label="Data Act Art. 3(2)">
2. Before concluding a contract for the purchase, rent or lease of a connected product, the seller, rentor or lessor, which may be the manufacturer, shall provide at least the following information to the user, in a clear and comprehensible manner:
(a) the type, format and estimated volume of product data which the connected product is capable of generating;
(b) whether the connected product is capable of generating data continuously and in real-time;
(c) whether the connected product is capable of storing data on-device or on a remote server, including, where applicable, the intended duration of retention;
(d) how the user may access, retrieve or, where relevant, erase the data, including the technical means to do so, as well as their terms of use and quality of service.
</source>
</cited_sources>
~~~~

Grades to give: correctness 1/2/3; faithful y/n; citations adequate y/n; note (optional).

### 24. q094-e983f3c63f1f [answer]

~~~~text
q094-e983f3c63f1f  [answer]
<question>
How long must deployers of high-risk AI systems keep the automatically generated logs?
</question>

<reference_answer>
For a period appropriate to the system's intended purpose and at least six months, to the extent the logs are under their control, unless other Union or national law provides otherwise.
</reference_answer>

<answer>
Status: answered

Deployers of high-risk AI systems must retain automatically generated logs for a period of at least six months, as specified in the AI Act (Article 26(6)). The exact retention period may vary depending on the intended purpose of the AI system, but it cannot be shorter than six months.

Claims:
1. Deployers of high-risk AI systems must retain automatically generated logs for a period of at least six months
   [AI Act Art. 26(6)] "Deployers of high-risk AI systems shall keep the logs automatically generated by that high-risk AI system to the extent such logs are under their control, for a period appropriate to the intended purpose of the high-risk AI system, of at least six months, unless provided otherwise in applicable Union or national law, in particular in Union law on the protection of personal data."

Caveats:
- General information from the EU legal texts, not legal advice.
</answer>

<cited_sources>
<source label="AI Act Art. 26(6)">
6. Deployers of high-risk AI systems shall keep the logs automatically generated by that high-risk AI system to the extent such logs are under their control, for a period appropriate to the intended purpose of the high-risk AI system, of at least six months, unless provided otherwise in applicable Union or national law, in particular in Union law on the protection of personal data.
Deployers that are financial institutions subject to requirements regarding their internal governance, arrangements or processes under Union financial services law shall maintain the logs as part of the documentation kept pursuant to the relevant Union financial service law.
</source>
</cited_sources>
~~~~

Grades to give: correctness 1/2/3; faithful y/n; citations adequate y/n; note (optional).

### 25. q094-5ff9598231ee [answer]

~~~~text
q094-5ff9598231ee  [answer]
<question>
How long must deployers of high-risk AI systems keep the automatically generated logs?
</question>

<reference_answer>
For a period appropriate to the system's intended purpose and at least six months, to the extent the logs are under their control, unless other Union or national law provides otherwise.
</reference_answer>

<answer>
Status: answered

Deployers of high-risk AI systems must retain automatically generated logs for a period of at least six months, as specified in the AI Act (Article 26(6)). The exact retention period may vary depending on the intended purpose of the AI system, but it cannot be shorter than six months.

Claims:
1. Deployers of high-risk AI systems must not retain automatically generated logs for a period of at least six months
   [AI Act Art. 26(6)] "Deployers of high-risk AI systems shall keep the logs automatically generated by that high-risk AI system to the extent such logs are under their control, for a period appropriate to the intended purpose of the high-risk AI system, of at least six months, unless provided otherwise in applicable Union or national law, in particular in Union law on the protection of personal data."

Caveats:
- General information from the EU legal texts, not legal advice.
</answer>

<cited_sources>
<source label="AI Act Art. 26(6)">
6. Deployers of high-risk AI systems shall keep the logs automatically generated by that high-risk AI system to the extent such logs are under their control, for a period appropriate to the intended purpose of the high-risk AI system, of at least six months, unless provided otherwise in applicable Union or national law, in particular in Union law on the protection of personal data.
Deployers that are financial institutions subject to requirements regarding their internal governance, arrangements or processes under Union financial services law shall maintain the logs as part of the documentation kept pursuant to the relevant Union financial service law.
</source>
</cited_sources>
~~~~

Grades to give: correctness 1/2/3; faithful y/n; citations adequate y/n; note (optional).

### 26. q135-f74d77de05e3 [refusal]

~~~~text
q135-f74d77de05e3  [refusal]
<question>
What strong customer authentication rules does PSD2 set for online payments?
</question>

<reference_answer>
(none)
</reference_answer>

<answer>
Status: refused (out_of_scope)

This question is about law or facts outside the EU texts I cover (GDPR, AI Act, DSA, NIS2, Data Act), so I can't answer it from my sources.
</answer>

<cited_sources>
(none)
</cited_sources>
~~~~

Grades to give: refusal appropriate y/n; note (optional).

### 27. q103-214d4945bde8 [answer]

~~~~text
q103-214d4945bde8  [answer]
<question>
Must online platforms prioritise notices from trusted flaggers?
</question>

<reference_answer>
Yes. Notices submitted by trusted flaggers within their area of expertise must be given priority and processed and decided upon without undue delay.
</reference_answer>

<answer>
Status: answered

Yes, online platforms must prioritize notices from trusted flaggers under the DSA (Article 22(1)). Providers of online platforms are required to take technical and organisational measures to ensure notices submitted by trusted flaggers are processed with priority and without undue delay.

Claims:
1. Online platforms must prioritize notices from trusted flaggers
   [DSA Art. 22(1)] "Providers of online platforms shall take the necessary technical and organisational measures to ensure that notices submitted by trusted flaggers, acting within their designated area of expertise, through the mechanisms referred to in Article 16, are given priority and are processed and decided upon without undue delay."

Caveats:
- General information from the EU legal texts, not legal advice.
</answer>

<cited_sources>
<source label="DSA Art. 22(1)">
1. Providers of online platforms shall take the necessary technical and organisational measures to ensure that notices submitted by trusted flaggers, acting within their designated area of expertise, through the mechanisms referred to in Article 16, are given priority and are processed and decided upon without undue delay.
</source>
</cited_sources>
~~~~

Grades to give: correctness 1/2/3; faithful y/n; citations adequate y/n; note (optional).

### 28. q071-6dcb7b159d1b [answer]

~~~~text
q071-6dcb7b159d1b  [answer]
<question>
Which safeguards allow international transfers without specific authorisation from a supervisory authority?
</question>

<reference_answer>
Legally binding instruments between public authorities, binding corporate rules, standard data protection clauses adopted by the Commission or by a supervisory authority and approved by the Commission, approved codes of conduct, and approved certification mechanisms, each with binding and enforceable commitments.
</reference_answer>

<answer>
Status: answered

Under GDPR Article 46(2), international data transfers may occur without specific supervisory authority authorisation if appropriate safeguards are met. These include six specific mechanisms: (a) legally binding instruments between public authorities, (b) binding corporate rules, (c) Commission-approved standard data protection clauses, (d) supervisory authority-approved clauses, (e) approved code of conduct with commitments, and (f) approved certification mechanisms with commitments.

Claims:
1. a legally binding and enforceable instrument between public authorities or bodies;
   [GDPR Art. 46(2)] "(a) a legally binding and enforceable instrument between public authorities or bodies;"
2. binding corporate rules in accordance with Article 47;
   [GDPR Art. 46(2)] "(b) binding corporate rules in accordance with Article 47;"
3. standard data protection clauses adopted by the Commission in accordance with the examination procedure referred to in Article 93(2);
   [GDPR Art. 46(2)] "(c) standard data protection clauses adopted by the Commission in accordance with the examination procedure referred to in Article 93(2);"
4. standard data protection clauses adopted by a supervisory authority and approved by the Commission pursuant to the examination procedure referred to in Article 93(2);
   [GDPR Art. 46(2)] "(d) standard data protection clauses adopted by a supervisory authority and approved by the Commission pursuant to the examination procedure referred to in Article 93(2);"
5. an approved code of conduct pursuant to Article 40 together with binding and enforceable commitments of the controller or processor in the third country to apply the appropriate safeguards, including as regards data subjects' rights;
   [GDPR Art. 46(2)] "(e) an approved code of conduct pursuant to Article 40 together with binding and enforceable commitments of the controller or processor in the third country to apply the appropriate safeguards, including as regards data subjects' rights;"
6. an approved certification mechanism pursuant to Article 42 together with binding and enforceable commitments of the controller or processor in the third country to apply the appropriate safeguards, including as regards data subjects' rights;
   [GDPR Art. 46(2)] "(f) an approved certification mechanism pursuant to Article 42 together with binding and enforceable commitments of the controller or processor in the third country to apply the appropriate safeguards, including as regards data subjects' rights;"

Caveats:
- General information from the EU legal texts, not legal advice.
</answer>

<cited_sources>
<source label="GDPR Art. 46(2)">
2. The appropriate safeguards referred to in paragraph 1 may be provided for, without requiring any specific authorisation from a supervisory authority, by:
(a) a legally binding and enforceable instrument between public authorities or bodies;
(b) binding corporate rules in accordance with Article 47;
(c) standard data protection clauses adopted by the Commission in accordance with the examination procedure referred to in Article 93(2);
(d) standard data protection clauses adopted by a supervisory authority and approved by the Commission pursuant to the examination procedure referred to in Article 93(2);
(e) an approved code of conduct pursuant to Article 40 together with binding and enforceable commitments of the controller or processor in the third country to apply the appropriate safeguards, including as regards data subjects' rights; or
(f) an approved certification mechanism pursuant to Article 42 together with binding and enforceable commitments of the controller or processor in the third country to apply the appropriate safeguards, including as regards data subjects' rights.
</source>
</cited_sources>
~~~~

Grades to give: correctness 1/2/3; faithful y/n; citations adequate y/n; note (optional).

### 29. q075-6a84a43e8d87 [answer]

~~~~text
q075-6a84a43e8d87  [answer]
<question>
From what date has the GDPR applied?
</question>

<reference_answer>
It has applied from 25 May 2018.
</reference_answer>

<answer>
Status: answered

The GDPR entered into force on 25 May 2018, as specified in Article 99(2).

Claims:
1. The GDPR applied from 25 May 2018
   [GDPR Art. 96] "International agreements involving the transfer of personal data to third countries or international organisations which were concluded by Member States prior to 24 May 2016, and which comply with Uni"

Caveats:
- General information from the EU legal texts, not legal advice.
</answer>

<cited_sources>
<source label="GDPR Art. 96">
International agreements involving the transfer of personal data to third countries or international organisations which were concluded by Member States prior to 24 May 2016, and which comply with Union law as applicable prior to that date, shall remain in force until amended, replaced or revoked.
</source>
</cited_sources>
~~~~

Grades to give: correctness 1/2/3; faithful y/n; citations adequate y/n; note (optional).

### 30. q017-4672ba631cbe [answer]

~~~~text
q017-4672ba631cbe  [answer]
<question>
What must a controller's record of processing activities contain?
</question>

<reference_answer>
The controller's name and contact details (and those of any joint controller, representative and DPO); the purposes of processing; the categories of data subjects and of personal data; the categories of recipients; transfers to third countries or international organisations with their identification and, where applicable, the safeguards; where possible, the time limits for erasure; and where possible, a general description of the technical and organisational security measures.
</reference_answer>

<answer>
Status: answered

Under GDPR Article 30(1), a controller's record of processing activities must contain the name and contact details of the controller, the purposes of processing, categories of data subjects and personal data, recipients, transfers to third countries, time limits for erasure, and technical/organisational security measures where possible.

Claims:
1. The record must contain the name and contact details of the controller and, where applicable, the joint controller, the controller's representative and the data protection officer.
   [GDPR Art. 30(1)] "(a) the name and contact details of the controller and, where applicable, the joint controller, the controller's representative and the data protection officer;"
2. The record must contain the purposes of the processing.
   [GDPR Art. 30(1)] "(b) the purposes of the processing;"
3. The record must contain a description of the categories of data subjects and of the categories of personal data.
   [GDPR Art. 30(1)] "(c) a description of the categories of data subjects and of the categories of personal data;"
4. The record must contain the categories of recipients to whom the personal data have been or will be disclosed including recipients in third countries or international organisations.
   [GDPR Art. 30(1)] "(d) the categories of recipients to whom the personal data have been or will be disclosed including recipients in third countries or international organisations;"
5. The record must contain, where applicable, transfers of personal data to a third country or an international organisation, including the identification of that third country or international organisation and, in the case of transfers referred to in the second subparagraph of Article 49(1), the documentation of suitable safeguards.
   [GDPR Art. 30(1)] "(e) where applicable, transfers of personal data to a third country or an international organisation, including the identification of that third country or international organisation and, in the case of transfers referred to in the second subparagraph of Article 49(1), the documentation of suitable safeguards;"
6. The record must contain, where possible, the envisaged time limits for erasure of the different categories of data.
   [GDPR Art. 30(1)] "(f) where possible, the envisaged time limits for erasure of the different categories of data;"
7. The record must contain, where possible, a general description of the technical and organisational security measures referred to in Article 32(1).
   [GDPR Art. 30(1)] "(g) where possible, a general description of the technical and organisational security measures referred to in Article 32(1)."

Caveats:
- General information from the EU legal texts, not legal advice.
</answer>

<cited_sources>
<source label="GDPR Art. 30(1)">
1. Each controller and, where applicable, the controller's representative, shall maintain a record of processing activities under its responsibility. That record shall contain all of the following information:
(a) the name and contact details of the controller and, where applicable, the joint controller, the controller's representative and the data protection officer;
(b) the purposes of the processing;
(c) a description of the categories of data subjects and of the categories of personal data;
(d) the categories of recipients to whom the personal data have been or will be disclosed including recipients in third countries or international organisations;
(e) where applicable, transfers of personal data to a third country or an international organisation, including the identification of that third country or international organisation and, in the case of transfers referred to in the second subparagraph of Article 49(1), the documentation of suitable safeguards;
(f) where possible, the envisaged time limits for erasure of the different categories of data;
(g) where possible, a general description of the technical and organisational security measures referred to in Article 32(1).
</source>
</cited_sources>
~~~~

Grades to give: correctness 1/2/3; faithful y/n; citations adequate y/n; note (optional).

## Answer sheet

One line per item, in this format:

~~~~text
q070-2588a10cead0: correctness=_ faithful=_ citations_adequate=_ note=
q077-a2b43907e4ec: correctness=_ faithful=_ citations_adequate=_ note=
q075-6ded796a6c55: correctness=_ faithful=_ citations_adequate=_ note=
q009-990637bfe073: correctness=_ faithful=_ citations_adequate=_ note=
q061-7ac906d2fb03: correctness=_ faithful=_ citations_adequate=_ note=
q051-877b71ffbbd0: refusal_appropriate=_ note=
q028-39476ade77d0: correctness=_ faithful=_ citations_adequate=_ note=
q002-7dfbe6a6524f: correctness=_ faithful=_ citations_adequate=_ note=
q077-ff1f1cf6a056: correctness=_ faithful=_ citations_adequate=_ note=
q002-2b6461bc24d1: correctness=_ faithful=_ citations_adequate=_ note=
q122-fd188c4f085e: correctness=_ faithful=_ citations_adequate=_ note=
q034-996d1046925d: refusal_appropriate=_ note=
q037-990c83cc753b: correctness=_ faithful=_ citations_adequate=_ note=
q061-57910fb079ff: correctness=_ faithful=_ citations_adequate=_ note=
q085-290a418b7875: correctness=_ faithful=_ citations_adequate=_ note=
q056-bdfe47e9cb47: refusal_appropriate=_ note=
q105-e33655cf6bdb: refusal_appropriate=_ note=
q104-241a08bf5dc6: correctness=_ faithful=_ citations_adequate=_ note=
q037-7ed822b3a443: correctness=_ faithful=_ citations_adequate=_ note=
q131-aeaffe3f0a9b: refusal_appropriate=_ note=
q025-88c6c309511d: correctness=_ faithful=_ citations_adequate=_ note=
q068-572028e06ee9: correctness=_ faithful=_ citations_adequate=_ note=
q122-fdc199efcbf2: correctness=_ faithful=_ citations_adequate=_ note=
q094-e983f3c63f1f: correctness=_ faithful=_ citations_adequate=_ note=
q094-5ff9598231ee: correctness=_ faithful=_ citations_adequate=_ note=
q135-f74d77de05e3: refusal_appropriate=_ note=
q103-214d4945bde8: correctness=_ faithful=_ citations_adequate=_ note=
q071-6dcb7b159d1b: correctness=_ faithful=_ citations_adequate=_ note=
q075-6a84a43e8d87: correctness=_ faithful=_ citations_adequate=_ note=
q017-4672ba631cbe: correctness=_ faithful=_ citations_adequate=_ note=
~~~~

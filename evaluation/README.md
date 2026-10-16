# Out-of-domain test questions

`off_topic_questions.csv` holds 60 questions that the assistant should **not** answer, used only to
evaluate the "I don't know" (out-of-domain) behaviour. They are **never used for training**.

**Provenance (disclosure):** these questions were **written by an AI assistant (Claude, Anthropic)** at
the student's request and reviewed by the student. They are not part of the HASH dataset and are not
real user questions. Because they were not collected from real users, they may be cleaner and more
"typical" than real off-topic messages, which could make out-of-domain detection look easier than
it is in practice. This is discussed in the report's limitations.

| type | count | what it tests |
|---|---|---|
| `unrelated` | 20 | questions with nothing to do with health (farming, sport, money, ...) |
| `health_not_sti` | 20 | health questions outside the STI domain (malaria, diabetes, first aid, ...) |
| `near_domain` | 20 | sexual and reproductive health questions the dataset does **not** cover (contraception, periods, pregnancy, fertility, clinic locations) |

Topics the HASH dataset does cover (e.g. condoms, safe sex, STI testing, HPV mentioned under general
STIs, UTI-vs-STI symptoms, genital discharge as an STI sign, sex during pregnancy) were deliberately
avoided, and the list was checked against the dataset's questions by keyword, so every question is
truly out of scope.
Half of each type is used to tune the threshold and the other half to test it (fixed seed).

# Out-of-domain test questions

`off_topic_questions.csv` holds 60 questions that the assistant should **not** answer, used only to
evaluate the "I don't know" (out-of-domain) behaviour. They are **never used for training**.

**Disclosure:** I generated these questions with an AI assistant (Claude, Anthropic) and reviewed them.
They are not from the HASH dataset and they are not real user messages. Real off-topic messages are
probably messier, so out-of-domain detection may look easier on these than it would be in practice
(see the limitations in the report).

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

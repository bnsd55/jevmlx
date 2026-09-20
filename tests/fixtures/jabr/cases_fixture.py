"""Minimal fixture mimicking jabr's bench/cases.py structure (3 primitives).

One choice task, one noul task, one score task — enough to test the AST
parser's mapping for all three question types. NOT the real case text.
"""

from dataclasses import dataclass
from typing import Union

from von.types import Choice, Question, Noul, Score


@dataclass
class Case:
    state: str
    expected: Union[str, bool, int]


@dataclass
class Task:
    id: str
    type: str
    question: Question
    cases: list[Case]


def support_department() -> Task:
    return Task(
        id="support_department",
        type="choice",
        question=Choice(
            instructions="Which team should handle this support message?",
            criteria={
                "billing": "Refunds, payments, invoices",
                "tech": "Bugs, errors, crashes",
                "other": "Anything else",
            },
        ),
        cases=[
            Case("I was charged twice on my card.", "billing"),
            Case("The iOS app crashes every time.", "tech"),
            Case("Just wanted to say great product.", "other"),
        ],
    )


def refund_eligible() -> Task:
    return Task(
        id="refund_eligible",
        type="noul",
        question=Noul(
            instructions="The customer is entitled to a refund under the 30-day guarantee",
        ),
        cases=[
            Case("I purchased five days ago and never activated it.", True),
            Case("I have been using this every day for eight months.", False),
        ],
    )


def frustration_level() -> Task:
    return Task(
        id="frustration_level",
        type="score",
        question=Score(
            instructions="How frustrated does the customer appear?",
            criteria=[
                "Calm",
                "Frustrated but civil",
                "Very angry",
            ],
        ),
        cases=[
            Case("Quick question about CSV export.", 0),
            Case("This is the third time the sync failed.", 1),
            Case("This is ABSURD. Fix it NOW.", 2),
        ],
    )


ALL_TASK_FUNCTIONS = [
    support_department,
    refund_eligible,
    frustration_level,
]

TASKS: list[Task] = [fn() for fn in ALL_TASK_FUNCTIONS]
TASK_IDS = [t.id for t in TASKS]

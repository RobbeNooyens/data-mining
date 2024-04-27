from copy import deepcopy
from dataclasses import dataclass
from typing import List


@dataclass
class Pattern:
    support: float
    confidence: float
    lift: float
    items_base: List[str]
    items_add: List[str]

    @classmethod
    def from_apyori(cls, apyori_pattern, support):
        return cls(support, apyori_pattern.confidence, apyori_pattern.lift, list(apyori_pattern.items_base), list(apyori_pattern.items_add))

    def __str__(self):
        return f"{int(self.support*100)}% - {int(self.confidence*100)}% - {int(self.lift*100)/100}\t- {self.items_base} => {self.items_add}"

    def __repr__(self):
        return self.__str__()


@dataclass
class MiningSetup:
    min_support: float
    min_confidence: float
    name: str
    age_bins: dict
    education_bins: dict
    working_hours_bins: dict
    english_bins: dict
    pregnant_mappings: dict
    marital_status_mappings: dict
    income_mappings: dict

    def with_minimum_support(self, min_support):
        copy = deepcopy(self)
        copy.min_support = min_support
        copy.name = f"min_support {min_support}"
        return copy

    def with_minimum_confidence(self, min_confidence):
        copy = deepcopy(self)
        copy.min_confidence = min_confidence
        copy.name = f"min_confidence {min_confidence}"
        return copy

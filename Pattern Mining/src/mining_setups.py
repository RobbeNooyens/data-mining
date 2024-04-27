from datastructures import MiningSetup

from bins import *
from mappings import *

mining_setups = [
    specific_bins := MiningSetup(
        min_support=0.1,
        min_confidence=0.8,
        name='Specific bins',
        age_bins=age_1,
        education_bins=education_1,
        working_hours_bins=working_hours_1,
        english_bins=english_1,
        pregnant_mappings=pregnant_1,
        marital_status_mappings=marital_status_1,
        income_mappings=income_1
    ),
    general_bins := MiningSetup(
        min_support=0.05,
        min_confidence=0.8,
        name='General bins',
        age_bins=age_2,
        education_bins=education_2,
        working_hours_bins=working_hours_2,
        english_bins=english_2,
        pregnant_mappings=pregnant_2,
        marital_status_mappings=marital_status_1,
        income_mappings=income_1
    ),

    # Test influence minimum support
    min_support_test := MiningSetup(
        min_support=1,
        min_confidence=0.5,
        name='min_support 1',
        age_bins=age_2,
        education_bins=education_2,
        working_hours_bins=working_hours_2,
        english_bins=english_2,
        pregnant_mappings=pregnant_2,
        marital_status_mappings=marital_status_1,
        income_mappings=income_1
    ),
    min_support_test.with_minimum_support(0.9),
    min_support_test.with_minimum_support(0.8),
    min_support_test.with_minimum_support(0.7),
    min_support_test.with_minimum_support(0.6),
    min_support_test.with_minimum_support(0.5),
    min_support_test.with_minimum_support(0.4),
    min_support_test.with_minimum_support(0.3),
    min_support_test.with_minimum_support(0.2),
    min_support_test.with_minimum_support(0.1),

    # Test influence minimum confidence
    min_support_test := MiningSetup(
        min_support=0.1,
        min_confidence=1,
        name='min_confidence 1',
        age_bins=age_2,
        education_bins=education_2,
        working_hours_bins=working_hours_2,
        english_bins=english_2,
        pregnant_mappings=pregnant_2,
        marital_status_mappings=marital_status_1,
        income_mappings=income_1
    ),
    min_support_test.with_minimum_confidence(0.9),
    min_support_test.with_minimum_confidence(0.8),
    min_support_test.with_minimum_confidence(0.7),
    min_support_test.with_minimum_confidence(0.6),
    min_support_test.with_minimum_confidence(0.5),
    min_support_test.with_minimum_confidence(0.4),
    min_support_test.with_minimum_confidence(0.3),
    min_support_test.with_minimum_confidence(0.2),
    min_support_test.with_minimum_confidence(0.1),

    # Find as much rules as possible with male or female
    MiningSetup(
        min_support=0.05,
        min_confidence=0.5,
        name='Genders',
        age_bins=age_2,
        education_bins=education_2,
        working_hours_bins=working_hours_2,
        english_bins=english_2,
        pregnant_mappings=pregnant_2,
        marital_status_mappings=marital_status_1,
        income_mappings=income_1
    ),
]
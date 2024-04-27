# Imports
from dataclasses import asdict
from typing import List, Tuple

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import apyori

from datastructures import Pattern, MiningSetup
from mining_setups import mining_setups


def main():
    # Read data
    df_original = pd.read_excel('assignment1_income_levels.xlsx')
    # Inspect data
    inspect_data(df_original)
    # Fill null values
    fill_null_values(df_original)
    # Mine patterns
    patterns_list = []
    for setup in mining_setups:
        df = df_original.copy()
        # Categorize data
        categorize_data(df, setup)
        # Find patterns
        result = mine_patterns(df, setup)
        # Save patterns in list
        patterns_list.append((setup.name, result))
    # Save patterns to excel with current date and time in filename
    save_patterns(patterns_list, f'output/result.xlsx')


def inspect_data(df):
    print(df.head())
    print(df.describe())
    print(df.info())

    def plot_counts(df_to_plot):
        plt.barh(df_to_plot.index, df_to_plot.values)
        plt.tight_layout()
        plt.show()
        plt.close()

    plot_counts(df['workclass'].value_counts())
    plot_counts(df['education'].value_counts())
    plot_counts(df['marital status'].value_counts())
    plot_counts(df['occupation'].value_counts())
    plot_counts(df['income'].value_counts())
    # Create box plot of age, education and working hours
    sns.boxplot(x='age', data=df)
    plt.show()
    plt.close()
    sns.boxplot(x='education', data=df)
    plt.show()
    plt.close()
    sns.boxplot(x='workinghours', data=df)
    plt.show()
    plt.close()
    # Plot proportional income by age
    income_by_age = df.groupby(['age', 'income']).size().unstack(fill_value=0)
    income_proportional = income_by_age.div(income_by_age.sum(axis=1), axis=0)
    ax = income_proportional.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='viridis')
    ax.set_xlabel("Age")
    ax.set_ylabel("Proportion of Income Levels")
    ax.set_title("Proportional Stacked Barchart for Income by Age")
    plt.show()
    # Plot proportional income by education
    income_by_education = df.groupby(['education', 'income']).size().unstack(fill_value=0)
    ax = income_by_education.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='viridis')
    ax.set_xlabel("Education")
    ax.set_ylabel("Income Levels")
    ax.set_title("Stacked Barchart for Income by Education")
    plt.show()
    # Plot proportional income by workinghours
    income_by_workinghours = df.groupby(['workinghours', 'income']).size().unstack(fill_value=0)
    ax = income_by_workinghours.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='viridis')
    ax.set_xlabel("Working Hours")
    ax.set_ylabel("Income Levels")
    ax.set_title("Stacked Barchart for Income by Working Hours")
    plt.show()


def fill_null_values(df):
    # Replace null with 0 for ability to speak english to get a scale 0-4
    df['ability to speak english'] = df['ability to speak english'].fillna(0).astype(int)
    # Replace null with N/A for gave birth this year
    df['gave birth this year'] = df['gave birth this year'].fillna('N/A')

def categorize_data(df, setup):
    bins_per_column = {
        'age': setup.age_bins,
        'education': setup.education_bins,
        'workinghours': setup.working_hours_bins,
        'ability to speak english': setup.english_bins,
    }
    replacements_per_column = {
        'gave birth this year': setup.pregnant_mappings,
        'marital status': setup.marital_status_mappings,
        'income': setup.income_mappings
    }

    for column, bins in bins_per_column.items():
        df[column] = pd.cut(df[column], bins=[-1, *bins.keys()], labels=bins.values())
    for column, replacements in replacements_per_column.items():
        df[column] = df[column].replace(replacements)

def map_apriori_to_patterns(patterns: List[apyori.RelationRecord]):
    output = []
    for pattern_set in patterns:
        if len(pattern_set.items) == 1:
            # Skip single item patterns
            continue
        output.extend([Pattern.from_apyori(p, pattern_set.support) for p in pattern_set.ordered_statistics])
    return output

def save_patterns(patterns_list: List[Tuple[str, List[Pattern]]], filename):
    dataframes = dict()
    for name, patterns in patterns_list:
        patterns_dicts = [asdict(pattern) for pattern in patterns]
        # Round all floats to 2 decimals
        for pattern in patterns_dicts:
            for key, value in pattern.items():
                if isinstance(value, float):
                    pattern[key] = round(value, 2)
        dataframes[name] = pd.DataFrame(patterns_dicts)
    # Save to one excel file, each in a different sheet
    with pd.ExcelWriter(filename) as writer:
        for name, df in dataframes.items():
            df.to_excel(writer, sheet_name=name, index=False)


def mine_patterns(df, setup: MiningSetup):
    transactions = df.apply(lambda x: x.dropna().tolist(), axis=1).tolist()
    ap = apyori.apriori(transactions, min_support=setup.min_support, min_confidence=setup.min_confidence)
    patterns = map_apriori_to_patterns(ap)
    # Remove all patterns that contain N/A
    patterns = [p for p in patterns if 'N/A' not in p.items_base and 'N/A' not in p.items_add]
    # Order by support, then by confidence
    patterns.sort(key=lambda x: (x.support, x.confidence), reverse=True)
    return patterns


if __name__ == '__main__':
    main()
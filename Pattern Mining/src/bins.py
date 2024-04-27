# Each bin is a dictionary where the key is the start of the bin and the value is the name of the bin

# ================ Age Bins =================
age_1 = {
    22: "Immediately after school",
    24: "Juniors after college",
    26: "Juniors after university",
    28: "Mediors",
    30: "Young seniors",
    34: "Seniors",
    55: "Experienced seniors",
    67: "Retiring seniors",
    100: "Retired"
}
age_2 = {
    30: "Juniors",
    56: "Seniors",
    100: "Retired"
}

# ================ Education Bins =================
education_1 = {
    1: "No school",
    11: "Elementary school",
    15: "High school without diploma",
    16: "High school diploma",
    18: "High school with extra",
    20: "Attended college",
    21: "Bachelors",
    23: "Masters or extra professional degree",
    24: "Doctorate"
}

education_2 = {
    15: "No diploma",
    20: "No higher education",
    24: "Higher education"
}

# ================ Working Hours Bins =================
working_hours_1 = {
    19: "Part-time",
    29: "Part-time and little more",
    39: "Almost full-time",
    40: "Full-time",
    49: "Full-time and little more",
    59: "Overtime",
    100: "Outliers"
}

working_hours_2 = {
    39: "Part-time",
    40: "Full-time",
    100: "Overtime",
}

# ================ English Bins =================
english_1 = {
    0: "Native",
    1: "Fluent",
    2: "Good",
    3: "Poor",
    4: "Not at all"
}

english_2 = {
    0: "N/A",
    4: "Not native"
}


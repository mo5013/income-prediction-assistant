import re
import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = 'models/best_model.joblib'

REQUIRED_FIELDS = [
    'age',
    'workclass',
    'education',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'hours_per_week',
    'native_country'
]


DEFAULT_VALUES = {
    'fnlwgt': 200000,
    'education_num': 13,
    'capital_gain': 0,
    'capital_loss': 0
}


def load_model():
    model = joblib.load(MODEL_PATH)
    return model


def extract_number(pattern, text):
    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        for group in match.groups():
            if group is not None:
                return int(group)

    return None


def parse_user_input(text):
    text_lower = text.lower()
    features = DEFAULT_VALUES.copy()

    features['age'] = extract_number(
        r'(\d+)[-\s]?year[-\s]?old|age\s*(\d+)',
        text_lower
    )

    if features['age'] is None:
        age_match = re.search(r'\b([1-9][0-9])\b', text_lower)
        if age_match:
            features['age'] = int(age_match.group(1))

    features['hours_per_week'] = extract_number(
        r'(\d+)\s*hours',
        text_lower
    )

    if 'private' in text_lower:
        features['workclass'] = 'Private'
    elif 'self-employed' in text_lower or 'self employed' in text_lower:
        features['workclass'] = 'Self-emp-not-inc'
    elif 'government' in text_lower or 'gov' in text_lower:
        features['workclass'] = 'State-gov'
    else:
        features['workclass'] = None

    if 'bachelor' in text_lower:
        features['education'] = 'Bachelors'
        features['education_num'] = 13
    elif 'master' in text_lower:
        features['education'] = 'Masters'
        features['education_num'] = 14
    elif 'doctorate' in text_lower or 'phd' in text_lower:
        features['education'] = 'Doctorate'
        features['education_num'] = 16
    elif 'high school' in text_lower or 'hs-grad' in text_lower:
        features['education'] = 'HS-grad'
        features['education_num'] = 9
    elif 'some college' in text_lower:
        features['education'] = 'Some-college'
        features['education_num'] = 10
    else:
        features['education'] = None

    if 'married' in text_lower:
        features['marital_status'] = 'Married-civ-spouse'
    elif 'divorced' in text_lower:
        features['marital_status'] = 'Divorced'
    elif 'single' in text_lower or 'never married' in text_lower:
        features['marital_status'] = 'Never-married'
    else:
        features['marital_status'] = None

    if 'manager' in text_lower or 'management' in text_lower:
        features['occupation'] = 'Exec-managerial'
    elif 'tech' in text_lower or 'it' in text_lower or 'computer' in text_lower:
        features['occupation'] = 'Tech-support'
    elif 'sales' in text_lower:
        features['occupation'] = 'Sales'
    elif 'clerical' in text_lower or 'admin' in text_lower:
        features['occupation'] = 'Adm-clerical'
    elif 'service' in text_lower:
        features['occupation'] = 'Other-service'
    else:
        features['occupation'] = None

    if 'husband' in text_lower:
        features['relationship'] = 'Husband'
    elif 'wife' in text_lower:
        features['relationship'] = 'Wife'
    elif 'own child' in text_lower:
        features['relationship'] = 'Own-child'
    elif 'not in family' in text_lower:
        features['relationship'] = 'Not-in-family'
    else:
        features['relationship'] = 'Not-in-family'

    if 'female' in text_lower or 'woman' in text_lower:
        features['sex'] = 'Female'
    elif 'male' in text_lower or 'man' in text_lower:
        features['sex'] = 'Male'
    else:
        features['sex'] = None

    if 'white' in text_lower:
        features['race'] = 'White'
    elif 'black' in text_lower:
        features['race'] = 'Black'
    elif 'asian' in text_lower:
        features['race'] = 'Asian-Pac-Islander'
    else:
        features['race'] = 'White'

    features['native_country'] = 'United-States'

    missing_fields = [
        field for field in REQUIRED_FIELDS
        if features.get(field) is None
    ]

    return features, missing_fields


def make_prediction(model, features):
    input_data = pd.DataFrame([features])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return prediction, probability


def generate_response(prediction, probability, features):
    income_label = '>50K' if prediction == 1 else '<=50K'
    confidence = probability if prediction == 1 else 1 - probability

    response = f"""
Based on the information provided, the model predicts that this person is likely to earn **{income_label} per year**.

Estimated confidence: **{confidence:.1%}**

Key factors considered include:
- Age: {features['age']}
- Education: {features['education']}
- Occupation: {features['occupation']}
- Hours per week: {features['hours_per_week']}
- Workclass: {features['workclass']}

Please note that this prediction is based on historical census-style data and should not be used as a final decision-making tool.
"""

    return response


def main():
    st.title('Income Prediction Assistant')
    st.write(
        'Ask a natural language question about a person, and the app will predict '
        'whether their income is likely to be above or below $50K per year.'
    )

    model = load_model()

    user_input = st.text_area(
        'Enter a question:',
        placeholder='Example: I am a 42-year-old male with a Masters degree working in management for 50 hours per week in the United States. I work in the private sector and I am married.'
    )

    if st.button('Predict Income'):
        if not user_input.strip():
            st.warning('Please enter a description first.')
            return

        features, missing_fields = parse_user_input(user_input)

        if missing_fields:
            st.warning('I need more information before making a prediction.')
            st.write('Missing fields:')
            st.write(missing_fields)
            return

        prediction, probability = make_prediction(model, features)
        response = generate_response(prediction, probability, features)

        st.subheader('Prediction Result')
        st.markdown(response)

        st.subheader('Parsed Features')
        st.dataframe(pd.DataFrame([features]))


if __name__ == '__main__':
    main()
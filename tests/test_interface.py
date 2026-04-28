from src.app import parse_user_input


def test_parse_user_input_extracts_expected_features():
    text = (
        'I am a 42-year-old male with a Masters degree working in management '
        'for 50 hours per week in the United States. I work in the private sector '
        'and I am married.'
    )

    features, missing_fields = parse_user_input(text)

    assert features['age'] == 42
    assert features['education'] == 'Masters'
    assert features['occupation'] == 'Exec-managerial'
    assert features['hours_per_week'] == 50
    assert features['workclass'] == 'Private'
    assert features['sex'] == 'Male'
    assert missing_fields == []


def test_parse_user_input_handles_incomplete_query():
    text = 'Can you predict my income?'

    features, missing_fields = parse_user_input(text)

    assert 'age' in missing_fields
    assert 'education' in missing_fields
    assert 'occupation' in missing_fields
    assert 'hours_per_week' in missing_fields
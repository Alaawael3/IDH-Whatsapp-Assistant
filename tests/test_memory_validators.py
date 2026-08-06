from datetime import date, timedelta

from app.services.memory import (
    compute_age,
    normalize_birth_date,
    validate_birth_date,
    validate_national_id,
)


def test_validate_national_id_valid():
    is_valid, reason = validate_national_id("29001011234567")
    assert is_valid is True
    assert reason is None


def test_validate_national_id_wrong_length():
    is_valid, reason = validate_national_id("12345")
    assert is_valid is False
    assert "14 digits" in reason


def test_validate_national_id_all_zeros():
    is_valid, reason = validate_national_id("0" * 14)
    assert is_valid is False
    assert "zeros" in reason


def test_validate_national_id_non_numeric():
    is_valid, reason = validate_national_id("2900101123456A")
    assert is_valid is False
    assert "digits only" in reason


def test_validate_national_id_missing():
    is_valid, reason = validate_national_id(None)
    assert is_valid is False
    assert reason == "missing"


def test_validate_birth_date_valid_iso():
    is_valid, parsed, reason = validate_birth_date("1998-05-14")
    assert is_valid is True
    assert parsed == date(1998, 5, 14)
    assert reason is None


def test_validate_birth_date_valid_slash_format():
    is_valid, parsed, reason = validate_birth_date("14/5/1998")
    # %d/%m/%Y should match this
    assert is_valid is True
    assert parsed == date(1998, 5, 14)


def test_validate_birth_date_future():
    future = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    is_valid, parsed, reason = validate_birth_date(future)
    assert is_valid is False
    assert "future" in reason


def test_validate_birth_date_unrealistic_age():
    is_valid, parsed, reason = validate_birth_date("1850-01-01")
    assert is_valid is False
    assert "unrealistic" in reason


def test_validate_birth_date_bad_format():
    is_valid, parsed, reason = validate_birth_date("not-a-date")
    assert is_valid is False
    assert reason == "invalid date format"


def test_compute_age_before_birthday_this_year():
    today = date.today()
    # A birthday later this year (or today) than "today" hasn't happened yet
    # relative to itself -- construct a birth date one day after today's
    # month/day so this year hasn't "arrived" yet, 30 years ago.
    birth_year = today.year - 30
    if today.month == 12 and today.day == 31:
        return  # skip edge case
    birth = date(birth_year, today.month, today.day) + timedelta(days=1)
    age = compute_age(birth)
    assert age == 29


def test_normalize_birth_date_roundtrip():
    is_valid, parsed, _ = validate_birth_date("14/5/1998")
    assert is_valid
    assert normalize_birth_date(parsed) == "1998-05-14"

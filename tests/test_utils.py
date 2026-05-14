import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils import clean_price, clean_mileage, parse_features, extract_make_model


class TestCleanPrice:
    def test_lacs(self):
        assert clean_price("PKR 23.9 lacs") == 2_390_000

    def test_crore(self):
        assert clean_price("PKR 3.5 crore") == 35_000_000

    def test_none(self):
        assert clean_price(None) is None

    def test_empty(self):
        assert clean_price("") is None

    def test_lacs_with_comma(self):
        assert clean_price("PKR 1,200,000") == 1_200_000

    def test_33_point_25_lacs(self):
        assert clean_price("PKR 33.25 lacs") == 3_325_000


class TestCleanMileage:
    def test_km_with_comma(self):
        assert clean_mileage("120,000 km") == 120_000

    def test_plain_number(self):
        assert clean_mileage("786 km") == 786

    def test_none(self):
        assert clean_mileage(None) is None


class TestParseFeatures:
    def test_python_list_string(self):
        raw = "['ABS', 'Air Bags', 'Air Conditioning']"
        result = parse_features(raw)
        assert "ABS" in result
        assert "Air Bags" in result
        assert len(result) == 3

    def test_empty(self):
        assert parse_features("") == []

    def test_none(self):
        assert parse_features(None) == []


class TestExtractMakeModel:
    def test_known_make(self):
        make, model, variant = extract_make_model("Toyota Corolla GLi 2022")
        assert make == "Toyota"
        assert model == "Corolla"

    def test_honda(self):
        make, model, _ = extract_make_model("Honda Civic EX 1995")
        assert make == "Honda"
        assert model == "Civic"

    def test_suzuki(self):
        make, model, _ = extract_make_model("Suzuki Alto VXL AGS 2022")
        assert make == "Suzuki"
        assert model == "Alto"

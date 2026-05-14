import sys
import os
import pytest
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model.pkl")
ENCODERS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "encoders.pkl")


@pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="model.pkl not found — run train.py first",
)
class TestMLPrediction:
    def setup_method(self):
        self.model = joblib.load(MODEL_PATH)
        self.encoders = joblib.load(ENCODERS_PATH)

    def _encode(self, col, val):
        enc = self.encoders[col]
        if val in enc.classes_:
            return enc.transform([val])[0]
        return 0

    def _predict(self, make, fuel, trans, body, province, assembly, year, mileage, engine):
        age = 2026 - year
        row = [
            self._encode("make", make),
            self._encode("fuel_type", fuel),
            self._encode("transmission", trans),
            self._encode("body_type", body),
            self._encode("province", province),
            self._encode("assembly", assembly),
            year, age, mileage, engine,
        ]
        return float(self.model.predict(np.array([row]))[0])

    def test_corolla_reasonable_range(self):
        price = self._predict("Toyota", "Petrol", "Automatic", "Sedan", "Lahore", "Local", 2020, 30000, 1800)
        assert 500_000 < price < 20_000_000, f"Unreasonable price: {price}"

    def test_alto_cheaper_than_prado(self):
        alto = self._predict("Suzuki", "Petrol", "Automatic", "Hatchback", "Islamabad", "Local", 2022, 20000, 660)
        prado = self._predict("Toyota", "Petrol", "Automatic", "SUV", "Islamabad", "Imported", 2022, 20000, 4000)
        assert alto < prado, "Alto should cost less than Prado"

    def test_higher_mileage_lower_price(self):
        low_km = self._predict("Honda", "Petrol", "Automatic", "Sedan", "Lahore", "Local", 2018, 10000, 1500)
        high_km = self._predict("Honda", "Petrol", "Automatic", "Sedan", "Lahore", "Local", 2018, 150000, 1500)
        assert low_km >= high_km, "Higher mileage should not predict higher price"

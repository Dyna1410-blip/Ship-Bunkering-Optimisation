# fuel_model.py

class DeshViraatFuelModel:

    def __init__(self):
        # Derived from your consumption dataset
        self.base = 25.51
        self.cubic = 0.01459

    def daily_consumption(self, speed_knots):
        """
        Fuel consumption in MT/day
        """
        return self.base + self.cubic * (speed_knots ** 3)

    def voyage_days(self, distance_nm, speed_knots):
        return distance_nm / (speed_knots * 24)

    def fuel_required(self, distance_nm, speed_knots):

        days = self.voyage_days(
            distance_nm,
            speed_knots
        )

        daily_cons = self.daily_consumption(
            speed_knots
        )

        return days * daily_cons

    def route_summary(
        self,
        distance_nm,
        speed_knots
    ):

        days = self.voyage_days(
            distance_nm,
            speed_knots
        )

        daily_cons = self.daily_consumption(
            speed_knots
        )

        fuel = self.fuel_required(
            distance_nm,
            speed_knots
        )

        return {
            "speed_knots": speed_knots,
            "voyage_days": round(days, 2),
            "daily_consumption_mt": round(daily_cons, 2),
            "fuel_required_mt": round(fuel, 2)
        }
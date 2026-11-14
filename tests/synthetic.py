
import numpy as np
import pandas as pd

CATS = {
    "Day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "Age_band_of_driver": ["18-30", "31-50", "Over 51", "Under 18", "Unknown"],
    "Drivers_gender": ["Male", "Female", "Unknown"],
    "Educational_level": ["Junior high school", "Elementary school", "High school", "Above high school", "Illiterate"],
    "Vehicle_driver_relation": ["Employee", "Owner", "Other"],
    "Driving_experience": ["1-2yr", "2-5yr", "5-10yr", "Above 10yr", "Below 1yr", "No Licence"],
    "Type_of_vehicle": ["Automobile", "Lorry (41?100Q)", "Other", "Public (13?45 seats)", "Taxi", "Motorcycle"],
    "Owner_of_vehicle": ["Owner", "Governmental", "Organization", "Other"],
    "Service_year_of_vehicle": ["Below 1yr", "1-2yr", "2-5yrs", "5-10yrs", "Above 10yr", "Unknown"],
    "Defect_of_vehicle": ["No defect", "5", "7"],
    "Area_accident_occured": ["Residential areas", "Office areas", "Recreational areas", "Industrial areas", "Other"],
    "Lanes_or_Medians": ["Two-way (divided with broken lines road marking)", "Undivided Two way", "One way", "Other"],
    "Road_allignment": ["Tangent road with flat terrain", "Gentle horizontal curve", "Steep grade downward with mountainous terrain"],
    "Types_of_Junction": ["Y Shape", "Crossing", "O Shape", "T Shape", "No junction", "Other"],
    "Road_surface_type": ["Asphalt roads", "Earth roads", "Gravel roads", "Other"],
    "Road_surface_conditions": ["Dry", "Wet or damp", "Snow", "Flood over 3cm. deep"],
    "Light_conditions": ["Daylight", "Darkness - lights lit", "Darkness - no lighting"],
    "Weather_conditions": ["Normal", "Raining", "Cloudy", "Windy", "Other"],
    "Type_of_collision": ["Collision with roadside objects", "Vehicle with vehicle collision", "Collision with pedestrians",
                          "Rollover", "Collision with animals", "Other"],
    "Vehicle_movement": ["Going straight", "Turnover", "Overtaking", "Reversing", "Other"],
    "Casualty_class": ["Driver or rider", "Pedestrian", "Passenger", "na"],
    "Casualty_gender": ["Male", "Female", "na"],
    "Age_band_of_casualty": ["18-30", "31-50", "Under 18", "Over 51", "na"],
    "Work_of_casuality": ["Driver", "Self-employed", "Employee", "Student", "Unemployed", "Other"],
    "Fitness_of_casuality": ["Normal", "Deaf", "Blind", "Other"],
    "Pedestrian_movement": ["Not a Pedestrian", "Crossing from driver's nearside", "Unknown or other"],
    "Cause_of_accident": ["Moving Backward", "Overtaking", "Changing lane to the left", "No distancing",
                          "Driving carelessly", "Overspeed", "Driving at high speed", "Other"],
}


def make_synthetic(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.choice(v, n) for c, v in CATS.items()})
    hours = rng.integers(0, 24, n)
    df.insert(0, "Time", [f"{h}:{m:02d}:00" for h, m in zip(hours, rng.integers(0, 60, n))])
    df["Number_of_vehicles_involved"] = rng.integers(1, 6, n)
    df["Number_of_casualties"] = rng.integers(1, 6, n)

    score = (
        0.5 * (df["Cause_of_accident"] == "Overspeed")
        + 0.4 * (df["Light_conditions"] == "Darkness - no lighting")
        + 0.3 * (df["Type_of_collision"] == "Collision with pedestrians")
        + 0.2 * df["Number_of_casualties"]
        + rng.normal(0, 0.6, n)
    )
    severity = np.where(score > 1.55, "Fatal injury", np.where(score > 0.95, "Serious Injury", "Slight Injury"))
    df["Accident_severity"] = severity

    # Inject missing values like the real data
    for col, frac in {"Defect_of_vehicle": 0.35, "Service_year_of_vehicle": 0.3, "Educational_level": 0.06,
                      "Type_of_vehicle": 0.07}.items():
        df.loc[rng.random(n) < frac, col] = np.nan
    return df

import pandas as pd


CONSTANT_SENSORS = [
    "sensor_1",
    "sensor_5",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]

SENSORS = [
    f"sensor_{i}"
    for i in range(1, 22)
    if f"sensor_{i}" not in CONSTANT_SENSORS
]


def build_inference_features(data):
    data = data.copy()

    data = data.rename(columns={
        "op_setting_1": "setting_1",
        "op_setting_2": "setting_2",
        "op_setting_3": "setting_3",
    })

    data = data.sort_values(
        ["unit_id", "cycle"]
    ).reset_index(drop=True)

    data = data.drop(columns=CONSTANT_SENSORS)

    frames = []

    for sensor in SENSORS:
        previous = data.groupby("unit_id")[sensor].shift(1)

        for window in [5, 10, 20]:
            rolling = (
                previous.groupby(data["unit_id"])
                .rolling(window, min_periods=1)
            )

            frames.append(pd.DataFrame({
                f"{sensor}_rolling_mean_{window}":
                    rolling.mean()
                    .reset_index(level=0, drop=True)
                    .reset_index(drop=True),

                f"{sensor}_rolling_std_{window}":
                    rolling.std()
                    .reset_index(level=0, drop=True)
                    .reset_index(drop=True),
            }))

    data = pd.concat(
        [data, pd.concat(frames, axis=1)],
        axis=1,
    )

    frames = []

    for sensor in SENSORS:
        group = data.groupby("unit_id")[sensor]

        frames.append(pd.DataFrame({
            f"{sensor}_delta_1":
                data[sensor] - group.shift(1),

            f"{sensor}_delta_5":
                data[sensor] - group.shift(5),
        }))

    data = pd.concat(
        [data, pd.concat(frames, axis=1)],
        axis=1,
    )

    return data
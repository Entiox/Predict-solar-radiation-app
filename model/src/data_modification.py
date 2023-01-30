from pandas import to_datetime
from column_names import time, time_sun_rise, time_sun_set, pressure, humidity, wind_direction, wind_speed, temperature

def convertTimeToSeconds(dt):
    seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
    return seconds

def convertSecondsToTime(seconds):
    hour = (int)(seconds / 3600)
    minute = (int)(seconds - (hour * 3600)) / 60
    second = seconds - (hour * 3600) - (minute * 60)
    return f"{hour}:{minute}:{second}"

def formatTime(data):
    data[time] = to_datetime(data[time], format="%H:%M:%S").apply(convertTimeToSeconds)
    data[time_sun_rise] = to_datetime(data[time_sun_rise], format="%H:%M:%S").apply(convertTimeToSeconds)
    data[time_sun_set] = to_datetime(data[time_sun_set], format="%H:%M:%S").apply(convertTimeToSeconds)
    return data

def formatData(data):
    data[temperature] = data[temperature].astype(int)
    data[pressure] = data[pressure].astype(float)
    data[humidity] = data[humidity].astype(int)
    data[wind_direction] = data[wind_direction].astype(float)
    data[wind_speed] = data[wind_speed].astype(float)
    data = formatTime(data)
    return data

if(__name__ == "__main__"):
    print("This is the module for preparing data of a dataset.")

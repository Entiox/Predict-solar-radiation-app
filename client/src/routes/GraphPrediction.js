import { ToggleButtonGroup, ToggleButton, Button, Spinner } from "react-bootstrap";
import { LineChart, CartesianGrid, XAxis, YAxis, Tooltip, Legend, Line} from "recharts"
import { useEffect, useState } from "react";
import TimePickerInput from "react-time-picker-input"
import times from "../params/Times"
import Select from "react-select"
import filterAndSort from "../utils/FilterAndSort"
import "../styles/GraphPrediction.css"

function GraphPrediction() {
    const [selected, setSelected] = useState(1)
    const [startTime, setStartTime] = useState("00:00")
    const [endTime, setEndTime] = useState("23:59")
    const [startTemperature, setStartTemperature] = useState(34)
    const [endTemperature, setEndTemperature] = useState(71)
    const [startPressure, setStartPressure] = useState(30.19)
    const [endPressure, setEndPressure] = useState(30.56)
    const [startHumidity, setStartHumidity] = useState(8)
    const [endHumidity, setEndHumidity] = useState(103)
    const [startWindDirection, setStartWindDirection] = useState(0.09)
    const [endWindDirection, setEndWindDirection] = useState(359.95)
    const [startWindSpeed, setStartWindSpeed] = useState(0.00)
    const [endWindSpeed, setEndWindSpeed] = useState(40.50)

    const [time, setTime] = useState("00:00")
    const [sunriseTime, setSunriseTime] = useState({ value: times[0].sunrise, label: times[0].sunrise})
    const [sunsetTime, setSunsetTime] = useState({ value: times[0].sunset, label: times[0].sunset})
    const [temperature, setTemperature] = useState(34)
    const [pressure, setPressure] = useState(30.19)
    const [humidity, setHumidity] = useState(8)
    const [windDirection, setWindDirection] = useState(0.09)
    const [windSpeed, setWindSpeed] = useState(0.00)

    const [loading, setLoading] = useState(false)
    const [data, setData] = useState(null)
    const [showNoResponseMessage, setShowNoResponseMessage] = useState(false)
    
    useEffect(() => {
        if(loading) {
            let inputParams = []
            switch(selected){
                case 1: 
                    if((temperature < 34 || temperature > 71) ||
                        (pressure < 30.18 || pressure > 30.56) || 
                        (humidity < 8 || humidity > 103) || 
                        (windDirection < 0.09 || windDirection > 359.95) || 
                        (windSpeed < 0 || windSpeed > 40.5)) {
                        alert("Invalid input.")
                        setLoading(false)
                        return
                    }
                    let startTimeMinutes = parseInt(startTime.slice(0, 2), 10) * 60 + parseInt(startTime.slice(3, 5), 10)
                    let endTimeMinutes = parseInt(endTime.slice(0, 2), 10) * 60 + parseInt(endTime.slice(3, 5), 10)
                    if(startTimeMinutes < endTimeMinutes) {
                        for(let i = startTimeMinutes; i <= endTimeMinutes; i++) {
                            inputParams.push(
                                { 
                                    "Time": `${Math.floor(i / 60)}:${i % 60}:00`,
                                    "Temperature": `${Math.round(temperature)}`,
                                    "Pressure": `${Math.round(pressure * 100) / 100}`,
                                    "Humidity": `${Math.round(humidity)}`,
                                    "WindDirection(Degrees)": `${Math.round(windDirection * 100) / 100}`,
                                    "Speed": `${Math.round(windSpeed * 100) / 100}`,
                                    "TimeSunRise": `${sunriseTime.value}:00`,
                                    "TimeSunSet": `${sunsetTime.value}:00` 
                                }
                            )
                        }
                    }
                    else {
                        for(let i = startTimeMinutes; i >= endTimeMinutes; i--) {
                            inputParams.push(
                                { 
                                    "Time": `${Math.floor(i / 60)}:${i % 60}:00`,
                                    "Temperature": `${Math.round(temperature)}`,
                                    "Pressure": `${Math.round(pressure * 100) / 100}`,
                                    "Humidity": `${Math.round(humidity)}`,
                                    "WindDirection(Degrees)": `${Math.round(windDirection * 100) / 100}`,
                                    "Speed": `${Math.round(windSpeed * 100) / 100}`,
                                    "TimeSunRise": `${sunriseTime.value}:00`,
                                    "TimeSunSet": `${sunsetTime.value}:00` 
                                }
                            )
                        }
                    }
                    break;
                case 2:
                    if((startTemperature < 34 || startTemperature > 71) ||
                        (endTemperature < 34 || endTemperature > 71) ||
                        (pressure < 30.18 || pressure > 30.56) || 
                        (humidity < 8 || humidity > 103) || 
                        (windDirection < 0.09 || windDirection > 359.95) || 
                        (windSpeed < 0 || windSpeed > 40.5)) {
                        alert("Invalid input.")
                        setLoading(false)
                        return
                    }
                    if(startTemperature < endTemperature) {
                        for(let i = Math.round(startTemperature); i <= Math.round(endTemperature); i++) {
                            inputParams.push(
                                { 
                                    "Time": `${time}:00`,
                                    "Temperature": `${i}`,
                                    "Pressure": `${Math.round(pressure * 100) / 100}`,
                                    "Humidity": `${Math.round(humidity)}`,
                                    "WindDirection(Degrees)": `${Math.round(windDirection * 100) / 100}`,
                                    "Speed": `${Math.round(windSpeed * 100) / 100}`,
                                    "TimeSunRise": `${sunriseTime.value}:00`,
                                    "TimeSunSet": `${sunsetTime.value}:00` 
                                }
                            )
                        }
                    }
                    else {
                        for(let i = Math.round(startTemperature); i >= Math.round(endTemperature); i--) {
                            inputParams.push(
                                { 
                                    "Time": `${time}:00`,
                                    "Temperature": `${i}`,
                                    "Pressure": `${Math.round(pressure * 100) / 100}`,
                                    "Humidity": `${Math.round(humidity)}`,
                                    "WindDirection(Degrees)": `${Math.round(windDirection * 100) / 100}`,
                                    "Speed": `${Math.round(windSpeed * 100) / 100}`,
                                    "TimeSunRise": `${sunriseTime.value}:00`,
                                    "TimeSunSet": `${sunsetTime.value}:00`  
                                }
                            )
                        }
                    }
                    break;
                case 3:
                    if((temperature < 34 || temperature > 71) ||
                        (startPressure < 30.18 || startPressure > 30.56) || 
                        (endPressure < 30.18 || endPressure > 30.56) || 
                        (humidity < 8 || humidity > 103) || 
                        (windDirection < 0.09 || windDirection > 359.95) || 
                        (windSpeed < 0 || windSpeed > 40.5)) {
                        alert("Invalid input.")
                        setLoading(false)
                        return
                    }
                    if(startPressure < endPressure) {
                        for(let i = Math.round(startPressure * 100) / 100; i <= Math.round(endPressure * 100) / 100; i = Math.round((i + 0.01) * 100) / 100) {
                            inputParams.push(
                                { 
                                    "Time": `${time}:00`,
                                    "Temperature": `${Math.round(temperature)}`,
                                    "Pressure": `${i}`,
                                    "Humidity": `${Math.round(humidity)}`,
                                    "WindDirection(Degrees)": `${Math.round(windDirection * 100) / 100}`,
                                    "Speed": `${Math.round(windSpeed * 100) / 100}`,
                                    "TimeSunRise": `${sunriseTime.value}:00`,
                                    "TimeSunSet": `${sunsetTime.value}:00` 
                                }
                            )
                        }
                    }
                    else {
                        for(let i = Math.round(startPressure * 100) / 100; i >= Math.round(endPressure * 100) / 100; i = Math.round((i - 0.01) * 100) / 100) {
                            inputParams.push(
                                { 
                                    "Time": `${time}:00`,
                                    "Temperature": `${Math.round(temperature)}`,
                                    "Pressure": `${i}`,
                                    "Humidity": `${Math.round(humidity)}`,
                                    "WindDirection(Degrees)": `${Math.round(windDirection * 100) / 100}`,
                                    "Speed": `${Math.round(windSpeed * 100) / 100}`,
                                    "TimeSunRise": `${sunriseTime.value}:00`,
                                    "TimeSunSet": `${sunsetTime.value}:00` 
                                }
                            )
                        }
                    }
                    break
                case 4:
                    if((temperature < 34 || temperature > 71) ||
                        (pressure < 30.18 || pressure > 30.56) || 
                        (startHumidity < 8 || startHumidity > 103) || 
                        (endHumidity < 8 || endHumidity > 103) || 
                        (windDirection < 0.09 || windDirection > 359.95) || 
                        (windSpeed < 0 || windSpeed > 40.5)) {
                        alert("Invalid input.")
                        setLoading(false)
                        return
                    }
                    if(startHumidity < endHumidity) {
                        for(let i = Math.round(startHumidity); i <= Math.round(endHumidity); i++) {
                            inputParams.push(
                                { 
                                    "Time": `${time}:00`,
                                    "Temperature": `${Math.round(temperature)}`,
                                    "Pressure": `${Math.round(pressure * 100) / 100}`,
                                    "Humidity": `${i}`,
                                    "WindDirection(Degrees)": `${Math.round(windDirection * 100) / 100}`,
                                    "Speed": `${Math.round(windSpeed * 100) / 100}`,
                                    "TimeSunRise": `${sunriseTime.value}:00`,
                                    "TimeSunSet": `${sunsetTime.value}:00` 
                                }
                            )
                        }
                    }
                    else {
                        for(let i = Math.round(startHumidity); i >= Math.round(endHumidity); i--) {
                            inputParams.push(
                                { 
                                    "Time": `${time}:00`,
                                    "Temperature": `${Math.round(temperature)}`,
                                    "Pressure": `${Math.round(pressure * 100) / 100}`,
                                    "Humidity": `${i}`,
                                    "WindDirection(Degrees)": `${Math.round(windDirection * 100) / 100}`,
                                    "Speed": `${Math.round(windSpeed * 100) / 100}`,
                                    "TimeSunRise": `${sunriseTime.value}:00`,
                                    "TimeSunSet": `${sunsetTime.value}:00` 
                                }
                            )
                        }
                    }
                    break
                case 5:
                    if((temperature < 34 || temperature > 71) ||
                        (pressure < 30.18 || pressure > 30.56) || 
                        (humidity < 8 || humidity > 103) || 
                        (startWindDirection < 0.09 || startWindDirection > 359.95) || 
                        (endWindDirection < 0.09 || endWindDirection > 359.95) || 
                        (windSpeed < 0 || windSpeed > 40.5)) {
                        alert("Invalid input.")
                        setLoading(false)
                        return
                    }
                    if(startWindDirection < endWindDirection) {
                        for(let i = Math.round(startWindDirection * 100) / 100; i <= Math.round(endWindDirection * 100) / 100; i = Math.round((i + 0.01) * 100) / 100) {
                            inputParams.push(
                                { 
                                    "Time": `${time}:00`,
                                    "Temperature": `${Math.round(temperature)}`,
                                    "Pressure": `${Math.round(pressure * 100) / 100}`,
                                    "Humidity": `${Math.round(humidity)}`,
                                    "WindDirection(Degrees)": `${i}`,
                                    "Speed": `${Math.round(windSpeed * 100) / 100}`,
                                    "TimeSunRise": `${sunriseTime.value}:00`,
                                    "TimeSunSet": `${sunsetTime.value}:00` 
                                }
                            )
                        }
                    }
                    else {
                        for(let i = Math.round(startWindDirection * 100) / 100; i >= Math.round(endWindDirection * 100) / 100; i = Math.round((i - 0.01) * 100) / 100) {
                            inputParams.push(
                                { 
                                    "Time": `${time}:00`,
                                    "Temperature": `${Math.round(temperature)}`,
                                    "Pressure": `${Math.round(pressure * 100) / 100}`,
                                    "Humidity": `${Math.round(humidity)}`,
                                    "WindDirection(Degrees)": `${i}`,
                                    "Speed": `${Math.round(windSpeed * 100) / 100}`,
                                    "TimeSunRise": `${sunriseTime.value}:00`,
                                    "TimeSunSet": `${sunsetTime.value}:00` 
                                }
                            )
                        }
                    }
                    break
                default:
                    if((temperature < 34 || temperature > 71) ||
                        (pressure < 30.18 || pressure > 30.56) || 
                        (humidity < 8 || humidity > 103) || 
                        (windDirection < 0.09 || windDirection > 359.95) || 
                        (startWindSpeed < 0 || startWindSpeed > 40.5) || 
                        (endWindSpeed < 0 || endWindSpeed > 40.5)) {
                        alert("Invalid input.")
                        setLoading(false)
                        return
                    }
                    if(startWindSpeed < endWindSpeed) {
                        for(let i = Math.round(startWindSpeed * 100) / 100; i <= Math.round(endWindSpeed * 100) / 100; i = Math.round((i + 0.01) * 100) / 100) {
                            inputParams.push(
                                { 
                                    "Time": `${time}:00`,
                                    "Temperature": `${Math.round(temperature)}`,
                                    "Pressure": `${Math.round(pressure * 100) / 100}`,
                                    "Humidity": `${Math.round(humidity)}`,
                                    "WindDirection(Degrees)": `${Math.round(windDirection * 100) / 100}`,
                                    "Speed": `${i}`,
                                    "TimeSunRise": `${sunriseTime.value}:00`,
                                    "TimeSunSet": `${sunsetTime.value}:00` 
                                }
                            )
                        }
                    }
                    else {
                        for(let i = Math.round(startWindSpeed * 100) / 100; i >= Math.round(endWindSpeed * 100) / 100; i = Math.round((i - 0.01) * 100) / 100) {
                            inputParams.push(
                                { 
                                    "Time": `${time}:00`,
                                    "Temperature": `${Math.round(temperature)}`,
                                    "Pressure": `${Math.round(pressure * 100) / 100}`,
                                    "Humidity": `${Math.round(humidity)}`,
                                    "WindDirection(Degrees)": `${Math.round(windDirection * 100) / 100}`,
                                    "Speed": `${i}`,
                                    "TimeSunRise": `${sunriseTime.value}:00`,
                                    "TimeSunSet": `${sunsetTime.value}:00` 
                                }
                            )
                        }
                    }
            }
            fetch("http://127.0.0.1:5000/multi_predict", {
                    method: "POST",
                    cache: "no-cache",
                    headers: {
                        "Content-type": "application/json"
                    },
                    body: JSON.stringify(inputParams)
                }
            )
            .then(response => response.json())
            .then(data => {
                    switch(selected) {
                        case 1:
                            setData(data.solar_radiations.map((value, index) => { return { "Altering Variable": inputParams[index].Time, "Solar Radiation": value.solar_radiation } }))
                            break
                        case 2:
                            setData(data.solar_radiations.map((value, index) => { return { "Altering Variable": inputParams[index].Temperature, "Solar Radiation": value.solar_radiation } }))
                            break
                        case 3:
                            setData(data.solar_radiations.map((value, index) => { return { "Altering Variable": inputParams[index].Pressure, "Solar Radiation": value.solar_radiation } }))
                            break
                        case 4:
                            setData(data.solar_radiations.map((value, index) => { return { "Altering Variable": inputParams[index].Humidity, "Solar Radiation": value.solar_radiation } }))
                            break
                        case 5:
                            setData(data.solar_radiations.map((value, index) => { return { "Altering Variable": inputParams[index]["WindDirection(Degrees)"], "Solar Radiation": value.solar_radiation } }))
                            break
                        default:
                            setData(data.solar_radiations.map((value, index) => { return { "Altering Variable": inputParams[index].Speed, "Solar Radiation": value.solar_radiation } }))
                            break
                    }
                    setLoading(false)
                }
            )
            .catch(() => {
                    setLoading(false)
                    setShowNoResponseMessage(true)
                }
            )
        } 
    }, [loading])

    return (
        <div className="main-block">
            <div>
                <h4 className="mb-4">Select altering parameter:</h4>
                <ToggleButtonGroup type="radio" className="d-inline-flex flex-wrap" name="range-param-select" value={selected} onChange={(e) => setSelected(e)}>
                    <ToggleButton id="time-range-toggle" variant="dark" value={1}>
                        Time
                    </ToggleButton>
                    <ToggleButton id="temperature-range-toggle" variant="dark" value={2}>
                        Temperature
                    </ToggleButton>
                    <ToggleButton id="pressure-range-toggle" variant="dark" value={3}>
                        Pressure
                    </ToggleButton>
                    <ToggleButton id="humidity-range-toggle" variant="dark" value={4}>
                        Humidity
                    </ToggleButton>
                    <ToggleButton id="wind-direction-range-toggle" variant="dark" value={5}>
                        Wind Direction
                    </ToggleButton>
                    <ToggleButton id="wind-speed-range-toggle" variant="dark" value={6}>
                        Wind Speed
                    </ToggleButton>
                </ToggleButtonGroup>
            </div>
            <div id="altering-element-placeholder">
                <div className="m-5">
                    { selected === 1 ? 
                        <>
                            <h5>Time:</h5>
                            <div className="m-2 d-flex flex-row align-items-center">
                                <TimePickerInput
                                    value={ startTime }
                                    onChange={ (value) => setStartTime(value) }
                                />
                                -
                                <TimePickerInput
                                    value={ endTime }
                                    onChange={ (value) => setEndTime(value) }
                                />
                            </div>
                        </>
                    : null }
                    { selected === 2 ? 
                        <>
                            <h5>Temperature: (°F)</h5>
                            <div className="my-4 d-flex flex-row align-items-center">
                                <input className="me-2" type="number" min="34" max="71" step="1" value={ startTemperature } onChange={ (event) => setStartTemperature(event.target.value) }/>
                                -
                                <input className="ms-2" type="number" min="34" max="71" step="1" value={ endTemperature } onChange={ (event) => setEndTemperature(event.target.value) }/>
                            </div>
                        </>
                    : null }
                    { selected === 3 ? 
                        <>
                            <h5>Pressure: (inches Hg)</h5>
                            <div className="my-4 d-flex flex-row align-items-center">
                                <input className="me-2" type="number" min="30.19" max="30.56" step="0.01" value={ startPressure } onChange={ (event) => setStartPressure(event.target.value) }/>
                                -
                                <input className="ms-2" type="number" min="30.19" max="30.56" step="0.01" value={ endPressure } onChange={ (event) => setEndPressure(event.target.value) }/>
                            </div>
                        </>
                    : null }
                    { selected === 4 ? 
                        <>
                            <h5>Humidity: (RH %)</h5>
                            <div className="my-4 d-flex flex-row align-items-center">
                                <input className="me-2" type="number" min="8" max="103" step="1" value={ startHumidity } onChange={ (event) => setStartHumidity(event.target.value) }/>
                                -
                                <input className="ms-2" type="number" min="8" max="103" step="1" value={ endHumidity } onChange={ (event) => setEndHumidity(event.target.value) }/>
                            </div>
                        </>
                    : null }
                    { selected === 5 ? 
                        <>
                            <h5>Wind Direction: (Degrees)</h5>
                            <div className="my-4 d-flex flex-row align-items-center">
                                <input className="me-2" type="number" min="0.09" max="359.95" step="0.01" value={ startWindDirection } onChange={ (event) => setStartWindDirection(event.target.value) }/>
                                -
                                <input className="ms-2" type="number" min="0.09" max="359.95" step="0.01" value={ endWindDirection } onChange={ (event) => setEndWindDirection(event.target.value) }/>
                            </div>
                        </>
                    : null }
                    { selected === 6 ? 
                        <>
                            <h5>Wind Speed: (MPH)</h5>
                            <div className="my-4 d-flex flex-row align-items-center">
                                <input className="me-2" type="number" min="0.00" max="40.50" step="0.01" value={ startWindSpeed } onChange={ (event) => setStartWindSpeed(event.target.value) }/>
                                -
                                <input className="ms-2" type="number" min="0.00" max="40.50" step="0.01" value={ endWindSpeed } onChange={ (event) => setEndWindSpeed(event.target.value) }/>
                            </div>
                        </>
                    : null }
                </div>
            </div>
            <div>
                <h4 className="mb-4">Constant parameters:</h4>
                <div className="d-inline-flex flex-row justify-content-center flex-wrap">
                    { selected !== 1 ? 
                        <div className="d-flex flex-column align-items-center mx-4 mb-3">
                            <h5>Time of day:</h5>
                            <div className="my-2 d-flex flex-row align-items-center">
                                <TimePickerInput
                                    value={ time }
                                    onChange={ (value) => setTime(value) }
                                />
                            </div>
                        </div>
                    : null }
                    <div className="d-flex flex-column align-items-center mx-4 mb-3">
                        <h5>Sunrise time:</h5>
                        <div className="my-2 d-flex flex-row align-items-center">
                            <Select 
                                options={ filterAndSort(times.map(time => { return { value: time.sunrise, label: time.sunrise } })) }
                                value={ sunriseTime }
                                onChange = { (currentSunrise) => {
                                        setSunriseTime(currentSunrise)  
                                        if(times.find(element => element.sunrise === currentSunrise.value && element.sunset === sunsetTime.value) === undefined) {
                                            let sunsetTime = times[times.findIndex(object => object.sunrise === currentSunrise.value)].sunset
                                            setSunsetTime( { value: sunsetTime, label: sunsetTime } )    
                                        }  
                                    }    
                                }
                            />
                        </div>
                    </div>
                    <div className="d-flex flex-column align-items-center mx-4 mb-3">
                        <h5>Sunset time:</h5>
                        <div className="my-2 d-flex flex-row align-items-center">
                            <Select 
                                options={ filterAndSort(times.map(time => { return { value: time.sunset, label: time.sunset } })) }
                                value={ sunsetTime }
                                onChange = { (currentSunset) => {
                                        setSunsetTime(currentSunset)  
                                        if(times.find(element => element.sunrise === sunriseTime.value && element.sunset === currentSunset.value) === undefined) {
                                            let sunriseTime = times[times.findIndex(object => object.sunset === currentSunset.value)].sunrise
                                            setSunriseTime( { value: sunriseTime, label: sunriseTime } )
                                        }       
                                    }    
                                }
                            />
                        </div>
                    </div>
                    { selected !== 2 ? 
                        <div className="d-flex flex-column align-items-center mx-4 mb-3">
                            <h5>Temperature: (°F)</h5>
                            <div className="my-3 d-flex flex-row align-items-center">
                                <input className="me-2" type="number" min="34" max="71" step="1" value={ temperature } onChange={ (event) => setTemperature(event.target.value) }/>
                            </div>
                        </div>
                    : null }
                    { selected !== 3 ? 
                        <div className="d-flex flex-column align-items-center mx-4 mb-3">
                            <h5>Pressure: (inches Hg)</h5>
                            <div className="my-3 d-flex flex-row align-items-center">
                                <input className="me-2" type="number" min="30.19" max="30.56" step="0.01" value={ pressure } onChange={ (event) => setPressure(event.target.value) }/>
                            </div>
                        </div>
                    : null }
                    { selected !== 4 ? 
                        <div className="d-flex flex-column align-items-center mx-4 mb-3">
                            <h5>Humidity: (RH %)</h5>
                            <div className="my-3 d-flex flex-row align-items-center">
                                <input className="me-2" type="number" min="8" max="103" step="1" value={ humidity } onChange={ (event) => setHumidity(event.target.value) }/>
                            </div>
                        </div>
                    : null }
                    { selected !== 5 ? 
                        <div className="d-flex flex-column align-items-center mx-4 mb-3">
                            <h5>Wind Direction: (Degrees)</h5>
                            <div className="my-3 d-flex flex-row align-items-center">
                                <input className="me-2" type="number" min="0.09" max="359.95" step="0.01" value={ windDirection } onChange={ (event) => setWindDirection(event.target.value) }/>
                            </div>
                        </div>
                    : null }
                    { selected !== 6 ? 
                        <div className="d-flex flex-column align-items-center mx-4 mb-3">
                            <h5>Wind Speed: (MPH)</h5>
                            <div className="my-3 d-flex flex-row align-items-center">
                                <input className="me-2" type="number" min="0.00" max="40.50" step="0.01" value={ windSpeed } onChange={ (event) => setWindSpeed(event.target.value) }/>
                            </div>
                        </div>
                    : null }
                </div>
            </div>
            <div className="m-4">
                <Button variant="outline-dark" onClick={ () => {
                            setData(null)
                            setLoading(true)
                            setShowNoResponseMessage(false)
                        }
                    }
                >
                    View graph
                </Button>
            </div>
            <div className="m-5">
                { loading && 
                    <Spinner animation="border" role="status" variant="dark">
                        <span className="visually-hidden">Loading...</span>
                    </Spinner> 
                }
                { data &&
                    <>
                        <LineChart width={1000} height={500} data={data}>
                            <CartesianGrid />
                            <XAxis dataKey="Altering Variable" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            <Line type="monotone" dataKey="Solar Radiation" stroke="#2227b3" />
                        </LineChart>
                    </> 
                }
                { showNoResponseMessage &&
                    <h4>No response</h4>
                }
            </div>
        </div>
    )
}

export default GraphPrediction
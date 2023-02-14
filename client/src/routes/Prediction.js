import TimePickerInput from "react-time-picker-input"
import InputRange from "react-input-range"
import { Form, Container, Row, Col, Button, Spinner } from "react-bootstrap"
import { useEffect, useState } from "react"
import Select from "react-select"
import times from "../params/Times"
import filterAndSort from "../utils/FilterAndSort"
import "../styles/Prediction.css"

function Prediction() {
    const [time, setTime] = useState("00:00")
    const [pressure, setPressure] = useState(30.19)
    const [temperature, setTemperature] = useState(34)
    const [humidity, setHumidity] = useState(8)
    const [windDirection, setWindDirection] = useState(0.09)
    const [windSpeed, setWindSpeed] = useState(0)
    const [sunriseTime, setSunriseTime] = useState({ value: times[0].sunrise, label: times[0].sunrise})
    const [sunsetTime, setSunsetTime] = useState({ value: times[0].sunset, label: times[0].sunset})
    const [loading, setLoading] = useState(false)
    const [data, setData] = useState(null)
    const [showNoResponseMessage, setShowNoResponseMessage] = useState(false)

    useEffect(() => {  
        if(loading) {
            let inputParams = { 
                "Time": `${time}:00`,
                "Temperature": `${temperature}`,
                "Pressure": `${pressure.toFixed(2)}`,
                "Humidity": `${humidity}`,
                "WindDirection(Degrees)": `${windDirection.toFixed(2)}`,
                "Speed": `${windSpeed.toFixed(2)}`,
                "TimeSunRise": `${sunriseTime.value}:00`,
                "TimeSunSet": `${sunsetTime.value}:00` 
            };
            fetch("http://127.0.0.1:5000/single_predict", {
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
                    setData(data)
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

    return(
        <div>
            <div className="prediction-block p-3">
                <Form>
                    <Container className="p-5" fluid>
                        <Row className="mb-5">
                            <div>
                                <h2>Enter parameters:</h2>
                            </div>
                        </Row>
                        <Row className="mb-5">
                            <Col xs={12} lg={4} className="d-flex justify-content-center">
                                <Form.Group controlId="timeOfDayInput" className="d-flex flex-column time-param mb-4">
                                    <Form.Label className="pb-3">Time of day:</Form.Label>
                                    <TimePickerInput
                                        value={ time }
                                        onChange={ (value) => setTime(value) }
                                    />
                                </Form.Group>
                            </Col>
                            <Col xs={12} lg={4} className="d-flex justify-content-center">
                                <Form.Group controlId="sunriseTimeInput" className="d-flex flex-column time-param mb-4">
                                    <Form.Label className="pb-3">Sunrise time:</Form.Label>
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
                                </Form.Group>
                            </Col>
                            <Col xs={12} lg={4} className="d-flex justify-content-center">
                                <Form.Group controlId="sunsetTimeInput" className="d-flex flex-column time-param mb-4">
                                    <Form.Label className="pb-3">Sunset time:</Form.Label>
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
                                </Form.Group>
                            </Col>
                        </Row>
                        <Row className="mb-5">
                            <Col xs={12} lg={6} className="d-flex justify-content-center">
                                <Form.Group controlId="temperatureInput" className="d-flex flex-column other-param mb-4">
                                    <Form.Label className="pb-3">Temperature: (°F)</Form.Label>
                                    <InputRange
                                        minValue={ 34 }
                                        maxValue={ 71 }
                                        step={ 1 }
                                        value={ temperature }
                                        onChange={ value => setTemperature(value) }
                                    />
                                </Form.Group>
                            </Col>
                            <Col xs={12} lg={6} className="d-flex justify-content-center">
                                <Form.Group controlId="pressureInput" className="d-flex flex-column other-param mb-4">
                                    <Form.Label className="pb-3">Pressure: (inches Hg)</Form.Label>
                                    <InputRange
                                        minValue={ 30.19 }
                                        maxValue={ 30.56 }
                                        step={ 0.01 }
                                        value={ pressure.toFixed(2) }
                                        onChange={ value => setPressure(value) }
                                    />
                                </Form.Group>
                            </Col>
                        </Row>
                        <Row className="mb-5">
                            <Col xs={12} lg={6} className="d-flex justify-content-center">
                                <Form.Group controlId="humidityInput" className="d-flex flex-column other-param mb-4">
                                    <Form.Label className="pb-3">Humidity: (RH %)</Form.Label>
                                    <InputRange
                                        minValue={ 8 }
                                        maxValue={ 103 }
                                        step={ 1 }
                                        value={ humidity }
                                        onChange={ value => setHumidity(value) }
                                    />
                                </Form.Group>
                            </Col>
                            <Col xs={12} lg={6} className="d-flex justify-content-center">
                                <Form.Group controlId="windDirectionInput" className="d-flex flex-column other-param mb-4">
                                    <Form.Label className="pb-3">Wind Direction: (Degrees)</Form.Label>
                                    <InputRange
                                        minValue={ 0.09 }
                                        maxValue={ 359.95 }
                                        step={ 0.01 }
                                        value={ windDirection.toFixed(2) }
                                        onChange={ value => setWindDirection(value) }
                                    />
                                </Form.Group>
                            </Col>
                        </Row>
                        <Row className="mb-5">
                            <Col xs={12} lg={6} className="d-flex justify-content-center">
                                <Form.Group controlId="speedInput" className="d-flex flex-column other-param mb-4">
                                    <Form.Label className="pb-3">Wind Speed (MPH):</Form.Label>
                                    <InputRange
                                        minValue={ 0.00 }
                                        maxValue={ 40.50 }
                                        step={ 0.01 }
                                        value={ windSpeed.toFixed(2) }
                                        onChange={ value => setWindSpeed(value) }
                                    />
                                </Form.Group>
                            </Col>
                        </Row>
                        <Row className="mb-5">
                            <Col xs={12} className="d-flex justify-content-center">
                                <Button 
                                    size="lg" 
                                    variant="outline-dark" 
                                    onClick={ () => {
                                            setData(null)
                                            setLoading(true)
                                            setShowNoResponseMessage(false)
                                        } 
                                    }
                                >
                                    Predict Radiation
                                </Button>
                            </Col>
                        </Row>
                    </Container>
                </Form>
                <div className="results-block">
                    <div className="d-flex justify-content-center">
                        <div className="d-flex flex-column">
                            { loading && 
                                <Spinner animation="border" role="status" variant="dark">
                                    <span className="visually-hidden">Loading...</span>
                                </Spinner> 
                            }
                            { data &&
                                <>
                                    <h4>Result:</h4>
                                    <span>{`${data.solar_radiation} (W/m²)`}</span>
                                </> 
                            }
                            { showNoResponseMessage &&
                                <h4>No response</h4>
                            }
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default Prediction
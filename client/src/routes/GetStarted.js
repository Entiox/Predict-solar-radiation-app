import Button from "react-bootstrap/Button"
import { Link } from "react-router-dom"
import "../styles/GetStarted.css"
import { singlePredictionRoute } from "../App"

function GetStarted() {
    return(
        <div className="get_started-block d-flex flex-column">
            <div className="get_started-block__heading">
                <h2>Here you can predict solar radiation depending on specific parameters and also draw dependency graphs</h2>
            </div>
            <Link to={ singlePredictionRoute }>
                <Button size="lg" variant="outline-dark">Get Started</Button>
            </Link>
        </div>
    )
}

export default GetStarted
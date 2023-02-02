import { Nav, Navbar } from "react-bootstrap"
import { NavLink, Outlet } from "react-router-dom"
import "../styles/Shared.css"
import { singlePredictionRoute, graphPredictionRoute } from "../App";

function Shared() {
    return(
      <>
        <nav>
            <Navbar fixed="top" expand="lg" variant="dark" className="p-2 ps-5 navbar">
                <Navbar.Brand className="nav-brand">Predict Solar Radiation</Navbar.Brand>
                <Navbar.Toggle aria-controls="navbar-nav" />
                <Navbar.Collapse className="navbar-nav">
                    <Nav>
                        <Nav.Item className="nav-item">
                            <Nav.Link as={ NavLink } to="/">Get Started</Nav.Link>
                        </Nav.Item>
                        <Nav.Item className="nav-item">
                            <Nav.Link as={ NavLink } to={ singlePredictionRoute }>Make Prediction</Nav.Link>
                        </Nav.Item>
                        <Nav.Item className="nav-item">
                            <Nav.Link as={ NavLink } to={ graphPredictionRoute }>Draw Graph</Nav.Link>
                        </Nav.Item>
                    </Nav>
                </Navbar.Collapse>
            </Navbar>
        </nav>
        <div className="outlet">
            <Outlet />  
        </div>
        <footer>
            <div className="footer-block">
                <div className="d-flex flex-column">
                    <span className="footer-text">App made by Dražen Antunović</span>
                    <span className="footer-text">It is a project as part of the course called Computer Science: Service and Data Analysis</span>
                    <span className="footer-text">from Faculty of Electrical Engineering, Computer Science and Information Technology, Osijek</span>
                </div>
            </div>
        </footer>
      </>  
    );
}

export default Shared
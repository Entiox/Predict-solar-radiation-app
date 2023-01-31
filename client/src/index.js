import React from "react"
import ReactDOM from "react-dom/client"
import "./styles/index.css"
import App from "./App"
import "bootstrap/dist/css/bootstrap.min.css"
import "react-input-range/lib/css/index.css"
import "react-time-picker-input/dist/components/TimeInput.css"

const root = ReactDOM.createRoot(document.getElementById("root"))
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)

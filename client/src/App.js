import { BrowserRouter, Routes, Route } from "react-router-dom";
import GetStarted from "./routes/GetStarted";
import Shared from "./routes/Shared";
import Prediction from "./routes/Prediction"
import GraphDrawing from "./routes/GraphDrawing";
import Error from "./routes/Error";
import "./styles/App.css";

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={ <Shared /> }>
            <Route index element={ <GetStarted /> }/>
            <Route path="/single_prediction" element={ <Prediction /> }/>
            <Route path="/graph" element={ <GraphDrawing /> }/>
          </Route>
          <Route path="*" element={ <Error /> }/>
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;

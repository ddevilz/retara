import { Routes, Route } from "react-router-dom";
import Hero from "./components/Hero";
import About from "./components/About";
import CaseStudies from "./components/CaseStudies";
import Products from "./pages/Products";

function Landing() {
  return (
    <>
      <Hero />
      <About />
      <CaseStudies />
    </>
  );
}

function App() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/products" element={<Products />} />
    </Routes>
  );
}

export default App;

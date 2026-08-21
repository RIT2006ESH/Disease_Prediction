import { BrowserRouter, Routes, Route, Link, useLocation } from "react-router-dom";
import Home from "./pages/Home";
import DiabetesForm from "./pages/DiabetesForm";
import CardioForm from "./pages/CardioForm";

function Nav() {
  const location = useLocation();
  const isActive = (path: string) => location.pathname === path;

  const linkClass = (path: string) =>
    `font-mono text-xs uppercase tracking-widest pb-1 border-b-2 transition-colors ${
      isActive(path)
        ? "border-clinical text-ink"
        : "border-transparent text-ink/50 hover:text-ink"
    }`;

  return (
    <nav className="border-b border-line bg-paper/95 backdrop-blur sticky top-0 z-10">
      <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
        <Link to="/" className="font-mono font-bold text-sm tracking-widest">
          CDX <span className="text-clinical">// RISK</span>
        </Link>
        <div className="flex gap-8">
          <Link to="/" className={linkClass("/")}>Home</Link>
          <Link to="/diabetes" className={linkClass("/diabetes")}>Diabetes</Link>
          <Link to="/cardio" className={linkClass("/cardio")}>Cardiovascular</Link>
        </div>
      </div>
    </nav>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Nav />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/diabetes" element={<DiabetesForm />} />
        <Route path="/cardio" element={<CardioForm />} />
      </Routes>
    </BrowserRouter>
  );
}
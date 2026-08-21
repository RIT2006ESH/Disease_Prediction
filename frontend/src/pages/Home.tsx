import { Link } from "react-router-dom";
import PulseLine from "../components/PulseLine";

export default function Home() {
  return (
    <div className="max-w-5xl mx-auto px-6 py-20">
      <p className="eyebrow mb-4">Early-Stage Screening · ML + Explainable AI</p>
      <h1 className="font-mono text-4xl md:text-5xl font-bold leading-tight max-w-2xl">
        Chronic disease risk,
        <br />
        read like a vitals chart.
      </h1>
      <PulseLine className="w-full max-w-xl h-12 my-8" />
      <p className="text-ink/60 max-w-lg mb-12">
        Enter clinical markers to get a statistical risk estimate for diabetes
        or cardiovascular disease — with the contributing factors shown, not
        just a verdict.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-px bg-line border border-line">
        <Link
          to="/diabetes"
          className="bg-white p-8 hover:bg-paper transition-colors group"
        >
          <p className="eyebrow mb-2">Screening · Diabetes</p>
          <h2 className="font-mono text-xl font-semibold mb-2">Diabetes Risk</h2>
          <p className="text-sm text-ink/60 mb-4">
            Glucose, HbA1c, BMI, and history-based risk assessment.
          </p>
          <span className="text-xs font-mono text-clinical group-hover:underline">
            Start assessment →
          </span>
        </Link>
        <Link
          to="/cardio"
          className="bg-white p-8 hover:bg-paper transition-colors group"
        >
          <p className="eyebrow mb-2">Screening · Cardiovascular</p>
          <h2 className="font-mono text-xl font-semibold mb-2">Cardiovascular Risk</h2>
          <p className="text-sm text-ink/60 mb-4">
            ECG, chest pain type, and exercise-response based risk assessment.
          </p>
          <span className="text-xs font-mono text-clinical group-hover:underline">
            Start assessment →
          </span>
        </Link>
      </div>

      <p className="text-xs text-ink/40 mt-10 font-mono">
        For educational/screening purposes only. Not a medical diagnosis.
      </p>
    </div>
  );
}
import { useState } from "react";
import { predictCardio } from "../api/client";
import type { CardioFormData, PredictionResponse } from "../types/prediction";
import ResultDisplay from "../components/ResultDisplay";
import PulseLine from "../components/PulseLine";

const initialForm: CardioFormData = {
  age: 50,
  sex: 1,
  cp: 0,
  trestbps: 120,
  chol: 200,
  fbs: 0,
  restecg: 0,
  thalach: 150,
  exang: 0,
  oldpeak: 0,
  slope: 1,
  ca: 0,
  thal: 3,
};

export default function CardioForm() {
  const [form, setForm] = useState<CardioFormData>(initialForm);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (field: keyof CardioFormData, value: number) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await predictCardio(form);
      setResult(response);
    } catch (err) {
      setError("Prediction failed. Check that the backend is running.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-xl mx-auto px-6 py-16">
      <p className="eyebrow mb-2">Screening · Cardiovascular</p>
      <h2 className="font-mono text-2xl font-bold mb-8">Cardiovascular Risk Assessment</h2>

      <form onSubmit={handleSubmit} className="space-y-5 card p-8">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="field-label">Age</label>
            <input
              type="number"
              className="field-input"
              value={form.age}
              onChange={(e) => handleChange("age", Number(e.target.value))}
            />
          </div>
          <div>
            <label className="field-label">Sex</label>
            <select
              className="field-input"
              value={form.sex}
              onChange={(e) => handleChange("sex", Number(e.target.value))}
            >
              <option value={1}>Male</option>
              <option value={0}>Female</option>
            </select>
          </div>
        </div>

        <div>
          <label className="field-label">Chest Pain Type</label>
          <select
            className="field-input"
            value={form.cp}
            onChange={(e) => handleChange("cp", Number(e.target.value))}
          >
            <option value={0}>Typical Angina</option>
            <option value={1}>Atypical Angina</option>
            <option value={2}>Non-anginal Pain</option>
            <option value={3}>Asymptomatic</option>
          </select>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="field-label">Resting BP (mm Hg)</label>
            <input
              type="number"
              className="field-input"
              value={form.trestbps}
              onChange={(e) => handleChange("trestbps", Number(e.target.value))}
            />
          </div>
          <div>
            <label className="field-label">Cholesterol (mg/dl)</label>
            <input
              type="number"
              className="field-input"
              value={form.chol}
              onChange={(e) => handleChange("chol", Number(e.target.value))}
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="field-label">Fasting Blood Sugar &gt;120</label>
            <select
              className="field-input"
              value={form.fbs}
              onChange={(e) => handleChange("fbs", Number(e.target.value))}
            >
              <option value={0}>No</option>
              <option value={1}>Yes</option>
            </select>
          </div>
          <div>
            <label className="field-label">Resting ECG</label>
            <select
              className="field-input"
              value={form.restecg}
              onChange={(e) => handleChange("restecg", Number(e.target.value))}
            >
              <option value={0}>Normal</option>
              <option value={1}>ST-T Abnormality</option>
              <option value={2}>LV Hypertrophy</option>
            </select>
          </div>
        </div>

        <div>
          <label className="field-label">Max Heart Rate Achieved</label>
          <input
            type="number"
            className="field-input"
            value={form.thalach}
            onChange={(e) => handleChange("thalach", Number(e.target.value))}
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="field-label">Exercise-Induced Angina</label>
            <select
              className="field-input"
              value={form.exang}
              onChange={(e) => handleChange("exang", Number(e.target.value))}
            >
              <option value={0}>No</option>
              <option value={1}>Yes</option>
            </select>
          </div>
          <div>
            <label className="field-label">ST Depression (oldpeak)</label>
            <input
              type="number"
              step="0.1"
              className="field-input"
              value={form.oldpeak}
              onChange={(e) => handleChange("oldpeak", Number(e.target.value))}
            />
          </div>
        </div>

        <div className="grid grid-cols-3 gap-4">
          <div>
            <label className="field-label">ST Slope</label>
            <select
              className="field-input"
              value={form.slope}
              onChange={(e) => handleChange("slope", Number(e.target.value))}
            >
              <option value={0}>Upsloping</option>
              <option value={1}>Flat</option>
              <option value={2}>Downsloping</option>
            </select>
          </div>
          <div>
            <label className="field-label">Major Vessels</label>
            <select
              className="field-input"
              value={form.ca}
              onChange={(e) => handleChange("ca", Number(e.target.value))}
            >
              <option value={0}>0</option>
              <option value={1}>1</option>
              <option value={2}>2</option>
              <option value={3}>3</option>
            </select>
          </div>
          <div>
            <label className="field-label">Thalassemia</label>
            <select
              className="field-input"
              value={form.thal}
              onChange={(e) => handleChange("thal", Number(e.target.value))}
            >
              <option value={3}>Normal</option>
              <option value={6}>Fixed Defect</option>
              <option value={7}>Reversible Defect</option>
            </select>
          </div>
        </div>

        <button type="submit" disabled={loading} className="btn-primary">
          {loading ? "Predicting..." : "Predict Cardiovascular Risk"}
        </button>
      </form>

      {loading && <PulseLine className="w-full h-8 mt-6" />}
      {error && <p className="text-alert text-sm font-mono mt-4">{error}</p>}
      {result && <ResultDisplay result={result} />}
    </div>
  );
}
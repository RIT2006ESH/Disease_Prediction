import { useState } from "react";
import { predictCardio } from "../api/client";
import type { CardioFormData, PredictionResponse } from "../types/prediction";
import ResultDisplay from "../components/ResultDisplay";

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
    <div className="max-w-xl mx-auto p-6">
      <h2 className="text-2xl font-bold mb-6">Cardiovascular Risk Assessment</h2>

      <form onSubmit={handleSubmit} className="space-y-4 bg-white p-6 rounded-lg shadow-sm border">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-1">Age</label>
            <input
              type="number"
              className="w-full border rounded px-3 py-2"
              value={form.age}
              onChange={(e) => handleChange("age", Number(e.target.value))}
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Sex</label>
            <select
              className="w-full border rounded px-3 py-2"
              value={form.sex}
              onChange={(e) => handleChange("sex", Number(e.target.value))}
            >
              <option value={1}>Male</option>
              <option value={0}>Female</option>
            </select>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Chest Pain Type</label>
          <select
            className="w-full border rounded px-3 py-2"
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
            <label className="block text-sm font-medium mb-1">Resting BP (mm Hg)</label>
            <input
              type="number"
              className="w-full border rounded px-3 py-2"
              value={form.trestbps}
              onChange={(e) => handleChange("trestbps", Number(e.target.value))}
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Cholesterol (mg/dl)</label>
            <input
              type="number"
              className="w-full border rounded px-3 py-2"
              value={form.chol}
              onChange={(e) => handleChange("chol", Number(e.target.value))}
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-1">Fasting Blood Sugar &gt;120</label>
            <select
              className="w-full border rounded px-3 py-2"
              value={form.fbs}
              onChange={(e) => handleChange("fbs", Number(e.target.value))}
            >
              <option value={0}>No</option>
              <option value={1}>Yes</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Resting ECG</label>
            <select
              className="w-full border rounded px-3 py-2"
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
          <label className="block text-sm font-medium mb-1">Max Heart Rate Achieved</label>
          <input
            type="number"
            className="w-full border rounded px-3 py-2"
            value={form.thalach}
            onChange={(e) => handleChange("thalach", Number(e.target.value))}
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-1">Exercise-Induced Angina</label>
            <select
              className="w-full border rounded px-3 py-2"
              value={form.exang}
              onChange={(e) => handleChange("exang", Number(e.target.value))}
            >
              <option value={0}>No</option>
              <option value={1}>Yes</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">ST Depression (oldpeak)</label>
            <input
              type="number"
              step="0.1"
              className="w-full border rounded px-3 py-2"
              value={form.oldpeak}
              onChange={(e) => handleChange("oldpeak", Number(e.target.value))}
            />
          </div>
        </div>

        <div className="grid grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium mb-1">ST Slope</label>
            <select
              className="w-full border rounded px-3 py-2"
              value={form.slope}
              onChange={(e) => handleChange("slope", Number(e.target.value))}
            >
              <option value={0}>Upsloping</option>
              <option value={1}>Flat</option>
              <option value={2}>Downsloping</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Major Vessels</label>
            <select
              className="w-full border rounded px-3 py-2"
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
            <label className="block text-sm font-medium mb-1">Thalassemia</label>
            <select
              className="w-full border rounded px-3 py-2"
              value={form.thal}
              onChange={(e) => handleChange("thal", Number(e.target.value))}
            >
              <option value={3}>Normal</option>
              <option value={6}>Fixed Defect</option>
              <option value={7}>Reversible Defect</option>
            </select>
          </div>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? "Predicting..." : "Predict Cardiovascular Risk"}
        </button>
      </form>

      {error && <p className="text-red-600 mt-4">{error}</p>}
      {result && <ResultDisplay result={result} />}
    </div>
  );
}
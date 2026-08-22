import { useState } from "react";
import { predictDiabetes } from "../api/client";
import type { DiabetesFormData, PredictionResponse } from "../types/prediction";
import ResultDisplay from "../components/ResultDisplay";
import PulseLine from "../components/PulseLine";

const initialForm: DiabetesFormData = {
  gender: "Female",
  age: 45,
  hypertension: 0,
  heart_disease: 0,
  smoking_history: "never",
  bmi: 25,
  hba1c_level: 5.5,
  blood_glucose_level: 100,
};

export default function DiabetesForm() {
  const [form, setForm] = useState<DiabetesFormData>(initialForm);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (field: keyof DiabetesFormData, value: string | number) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await predictDiabetes(form);
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
      <p className="eyebrow mb-2">Screening · Diabetes</p>
      <h2 className="font-mono text-2xl font-bold mb-8">Diabetes Risk Assessment</h2>

      <form onSubmit={handleSubmit} className="space-y-5 card p-8">
        <div>
          <label className="field-label">Gender</label>
          <select
            className="field-input"
            value={form.gender}
            onChange={(e) => handleChange("gender", e.target.value)}
          >
            <option value="Female">Female</option>
            <option value="Male">Male</option>
            <option value="Other">Other</option>
          </select>
        </div>

        <div>
          <label className="field-label">Age</label>
          <input
            type="number"
            className="field-input"
            value={form.age}
            onChange={(e) => handleChange("age", Number(e.target.value))}
            min={0}
            max={120}
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="field-label">Hypertension</label>
            <select
              className="field-input"
              value={form.hypertension}
              onChange={(e) => handleChange("hypertension", Number(e.target.value) as 0 | 1)}
            >
              <option value={0}>No</option>
              <option value={1}>Yes</option>
            </select>
          </div>
          <div>
            <label className="field-label">Heart Disease</label>
            <select
              className="field-input"
              value={form.heart_disease}
              onChange={(e) => handleChange("heart_disease", Number(e.target.value) as 0 | 1)}
            >
              <option value={0}>No</option>
              <option value={1}>Yes</option>
            </select>
          </div>
        </div>

        <div>
          <label className="field-label">Smoking History</label>
          <select
            className="field-input"
            value={form.smoking_history}
            onChange={(e) => handleChange("smoking_history", e.target.value)}
          >
            <option value="never">Never</option>
            <option value="former">Former</option>
            <option value="current">Current</option>
            <option value="not current">Not Current</option>
            <option value="ever">Ever</option>
            <option value="No Info">No Info</option>
          </select>
        </div>

        <div>
          <label className="field-label">BMI</label>
          <input
            type="number"
            step="0.1"
            className="field-input"
            value={form.bmi}
            onChange={(e) => handleChange("bmi", Number(e.target.value))}
          />
        </div>

        <div>
          <label className="field-label">HbA1c Level (%)</label>
          <input
            type="number"
            step="0.1"
            className="field-input"
            value={form.hba1c_level}
            onChange={(e) => handleChange("hba1c_level", Number(e.target.value))}
          />
        </div>

        <div>
          <label className="field-label">Blood Glucose Level (mg/dL)</label>
          <input
            type="number"
            className="field-input"
            value={form.blood_glucose_level}
            onChange={(e) => handleChange("blood_glucose_level", Number(e.target.value))}
          />
        </div>

        <button type="submit" disabled={loading} className="btn-primary">
          {loading ? "Predicting..." : "Predict Diabetes Risk"}
        </button>
      </form>

      {loading && <PulseLine className="w-full h-8 mt-6" />}
      {error && <p className="text-alert text-sm font-mono mt-4">{error}</p>}
      {result && <ResultDisplay result={result} />}
    </div>
  );
}
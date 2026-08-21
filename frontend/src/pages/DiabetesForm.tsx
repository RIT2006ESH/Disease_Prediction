import { useState } from "react";
import { predictDiabetes } from "../api/client";
import type { DiabetesFormData, PredictionResponse } from "../types/prediction";
import ResultDisplay from "../components/ResultDisplay";

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
    <div className="max-w-xl mx-auto p-6">
      <h2 className="text-2xl font-bold mb-6">Diabetes Risk Assessment</h2>

      <form onSubmit={handleSubmit} className="space-y-4 bg-white p-6 rounded-lg shadow-sm border">
        <div>
          <label className="block text-sm font-medium mb-1">Gender</label>
          <select
            className="w-full border rounded px-3 py-2"
            value={form.gender}
            onChange={(e) => handleChange("gender", e.target.value)}
          >
            <option value="Female">Female</option>
            <option value="Male">Male</option>
            <option value="Other">Other</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Age</label>
          <input
            type="number"
            className="w-full border rounded px-3 py-2"
            value={form.age}
            onChange={(e) => handleChange("age", Number(e.target.value))}
            min={0}
            max={120}
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-1">Hypertension</label>
            <select
              className="w-full border rounded px-3 py-2"
              value={form.hypertension}
              onChange={(e) => handleChange("hypertension", Number(e.target.value) as 0 | 1)}
            >
              <option value={0}>No</option>
              <option value={1}>Yes</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Heart Disease</label>
            <select
              className="w-full border rounded px-3 py-2"
              value={form.heart_disease}
              onChange={(e) => handleChange("heart_disease", Number(e.target.value) as 0 | 1)}
            >
              <option value={0}>No</option>
              <option value={1}>Yes</option>
            </select>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Smoking History</label>
          <select
            className="w-full border rounded px-3 py-2"
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
          <label className="block text-sm font-medium mb-1">BMI</label>
          <input
            type="number"
            step="0.1"
            className="w-full border rounded px-3 py-2"
            value={form.bmi}
            onChange={(e) => handleChange("bmi", Number(e.target.value))}
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">HbA1c Level (%)</label>
          <input
            type="number"
            step="0.1"
            className="w-full border rounded px-3 py-2"
            value={form.hba1c_level}
            onChange={(e) => handleChange("hba1c_level", Number(e.target.value))}
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Blood Glucose Level (mg/dL)</label>
          <input
            type="number"
            className="w-full border rounded px-3 py-2"
            value={form.blood_glucose_level}
            onChange={(e) => handleChange("blood_glucose_level", Number(e.target.value))}
          />
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? "Predicting..." : "Predict Diabetes Risk"}
        </button>
      </form>

      {error && <p className="text-red-600 mt-4">{error}</p>}
      {result && <ResultDisplay result={result} />}
    </div>
  );
}
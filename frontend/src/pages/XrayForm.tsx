import { useState } from "react";
import { predictXray } from "../api/client";
import type { XrayPredictionResponse } from "../types/prediction";
import PulseLine from "../components/PulseLine";

export default function XrayForm() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<XrayPredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0] ?? null;
    setFile(selected);
    setResult(null);
    setError(null);
    setPreview(selected ? URL.createObjectURL(selected) : null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await predictXray(file);
      setResult(response);
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      setError(detail || "Prediction failed. Check that the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const isHighRisk = result?.risk_label === "High Risk";
  const riskColor = isHighRisk ? "text-alert" : "text-signal";
  const barColor = isHighRisk ? "bg-alert" : "bg-signal";

  return (
    <div className="max-w-xl mx-auto px-6 py-16">
      <p className="eyebrow mb-2">Screening · Pneumonia (Chest X-Ray)</p>
      <h2 className="font-mono text-2xl font-bold mb-8">X-Ray Risk Assessment</h2>

      <form onSubmit={handleSubmit} className="space-y-5 card p-8">
        <div>
          <label className="field-label">Chest X-Ray Image</label>
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="field-input file:mr-3 file:py-1 file:px-3 file:border-0 file:bg-ink file:text-paper file:text-xs file:font-mono file:uppercase"
          />
        </div>

        {preview && (
          <div className="border border-line p-2">
            <img src={preview} alt="Preview" className="w-full max-h-64 object-contain" />
          </div>
        )}

        <button type="submit" disabled={loading || !file} className="btn-primary">
          {loading ? "Analyzing..." : "Predict Pneumonia Risk"}
        </button>
      </form>

      {loading && <PulseLine className="w-full h-8 mt-6" />}
      {error && <p className="text-alert text-sm font-mono mt-4">{error}</p>}

      {result && (
        <div className="card mt-8">
          <div className="border-b border-line px-6 py-4 flex items-center justify-between">
            <p className="eyebrow">Result</p>
            <span className={`font-mono text-xs uppercase tracking-widest px-2 py-1 border ${
              isHighRisk ? "border-alert text-alert" : "border-signal text-signal"
            }`}>
              {result.risk_label}
            </span>
          </div>
          <div className="px-6 py-6">
            <div className="flex items-baseline gap-2 mb-2">
              <span className={`font-mono text-4xl font-bold ${riskColor}`}>
                {(result.probability * 100).toFixed(1)}%
              </span>
              <span className="text-xs text-ink/50 font-mono uppercase tracking-wide">
                pneumonia probability
              </span>
            </div>
            <div className="w-full h-1.5 bg-line mb-6">
              <div className={`h-1.5 ${barColor}`} style={{ width: `${result.probability * 100}%` }} />
            </div>
            <p className="text-xs text-ink/50 leading-relaxed">{result.disclaimer}</p>
            <p className="text-xs text-ink/30 font-mono mt-2">model · {result.model_version}</p>
          </div>
        </div>
      )}
    </div>
  );
}

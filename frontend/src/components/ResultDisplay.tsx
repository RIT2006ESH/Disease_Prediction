import type { PredictionResponse } from "../types/prediction";

interface Props {
  result: PredictionResponse;
}

export default function ResultDisplay({ result }: Props) {
  const isHighRisk = result.risk_label === "High Risk";
  const probabilityPct = (result.probability * 100).toFixed(1);
  const riskColor = isHighRisk ? "text-alert" : "text-signal";
  const barColor = isHighRisk ? "bg-alert" : "bg-signal";

  return (
    <div className="card mt-8 overflow-hidden">
      <div className="border-b border-line px-6 py-4 flex items-center justify-between">
        <h3 className="text-base font-semibold">Prediction Result</h3>
        <span className={`text-sm font-medium px-3 py-1 rounded-full ${
          isHighRisk ? "bg-alert/10 text-alert" : "bg-signal/10 text-signal"
        }`}>
          {result.risk_label}
        </span>
      </div>

      <div className="px-6 py-5">
        <div className="flex justify-between items-baseline mb-1.5">
          <span className="text-sm text-ink/60">Risk Probability</span>
          <span className={`text-lg font-semibold ${riskColor}`}>{probabilityPct}%</span>
        </div>
        <div className="w-full h-2 bg-line rounded-full mb-6">
          <div className={`h-2 rounded-full ${barColor}`} style={{ width: `${probabilityPct}%` }} />
        </div>

        <h4 className="text-sm font-semibold text-ink/80 mb-2">Top Contributing Factors</h4>
        <ul className="divide-y divide-line mb-5">
          {result.top_features.map((f) => (
            <li key={f.feature} className="flex justify-between py-2 text-sm">
              <span className="text-ink/70">{f.feature.replace(/^(num|cat|bin)__/, "")}</span>
              <span className={f.impact >= 0 ? "text-alert" : "text-signal"}>
                {f.impact >= 0 ? "+" : ""}
                {f.impact.toFixed(3)}
              </span>
            </li>
          ))}
        </ul>

        <p className="text-xs text-ink/50 leading-relaxed border-t border-line pt-3">
          {result.disclaimer}
        </p>
        <p className="text-xs text-ink/30 mt-1.5">Model: {result.model_version}</p>
      </div>
    </div>
  );
}
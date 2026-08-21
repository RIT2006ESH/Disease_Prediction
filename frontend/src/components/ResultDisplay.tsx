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
          <span className={`font-mono text-4xl font-bold ${riskColor}`}>{probabilityPct}%</span>
          <span className="text-xs text-ink/50 font-mono uppercase tracking-wide">probability</span>
        </div>
        <div className="w-full h-1.5 bg-line mb-6">
          <div className={`h-1.5 ${barColor}`} style={{ width: `${probabilityPct}%` }} />
        </div>

        <p className="eyebrow mb-3">Contributing Factors</p>
        <ul className="divide-y divide-line border-y border-line mb-6">
          {result.top_features.map((f) => (
            <li key={f.feature} className="flex justify-between py-2 text-sm">
              <span className="text-ink/70">{f.feature.replace(/^(num|cat|bin)__/, "")}</span>
              <span className={`font-mono ${f.impact >= 0 ? "text-alert" : "text-signal"}`}>
                {f.impact >= 0 ? "+" : ""}
                {f.impact.toFixed(3)}
              </span>
            </li>
          ))}
        </ul>

        <p className="text-xs text-ink/50 leading-relaxed">{result.disclaimer}</p>
        <p className="text-xs text-ink/30 font-mono mt-2">model · {result.model_version}</p>
      </div>
    </div>
  );
}
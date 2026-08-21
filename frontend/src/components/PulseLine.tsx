interface Props {
  className?: string;
  animate?: boolean;
}

export default function PulseLine({ className = "", animate = true }: Props) {
  return (
    <svg
      viewBox="0 0 400 60"
      className={className}
      fill="none"
      preserveAspectRatio="none"
    >
      <path
        d="M0 30 L100 30 L115 10 L130 50 L145 30 L165 30 L180 5 L195 55 L210 30 L400 30"
        stroke="#0B6E5C"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className={animate ? "pulse-line" : ""}
      />
    </svg>
  );
}
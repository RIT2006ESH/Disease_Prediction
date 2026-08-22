export interface FeatureContribution {
  feature: string;
  impact: number;
}

export interface PredictionResponse {
  risk_label: "Low Risk" | "High Risk";
  probability: number;
  top_features: FeatureContribution[];
  model_version: string;
  disclaimer: string;
}

export interface DiabetesFormData {
  gender: "Male" | "Female" | "Other";
  age: number;
  hypertension: 0 | 1;
  heart_disease: 0 | 1;
  smoking_history: "never" | "former" | "current" | "not current" | "ever" | "No Info";
  bmi: number;
  hba1c_level: number;
  blood_glucose_level: number;
}

export interface CardioFormData {
  age: number;
  sex: 0 | 1;
  cp: 0 | 1 | 2 | 3;
  trestbps: number;
  chol: number;
  fbs: 0 | 1;
  restecg: 0 | 1 | 2;
  thalach: number;
  exang: 0 | 1;
  oldpeak: number;
  slope: 0 | 1 | 2;
  ca: 0 | 1 | 2 | 3;
  thal: 3 | 6 | 7;
}
export interface XrayPredictionResponse {
  risk_label: "Low Risk" | "High Risk" | "Invalid Input";
  probability: number;
  model_version: string;
  valid_input: boolean;
  disclaimer: string;
}

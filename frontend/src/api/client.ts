import axios from "axios";
import type {
  DiabetesFormData,
  CardioFormData,
  PredictionResponse,
  XrayPredictionResponse,
} from "../types/prediction";

const API_BASE_URL = "http://localhost:8000/api/v1";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: { "Content-Type": "application/json" },
});

export async function predictDiabetes(data: DiabetesFormData): Promise<PredictionResponse> {
  const response = await apiClient.post<PredictionResponse>("/predict/diabetes", data);
  return response.data;
}

export async function predictCardio(data: CardioFormData): Promise<PredictionResponse> {
  const response = await apiClient.post<PredictionResponse>("/predict/cardio", data);
  return response.data;
}

export async function predictXray(file: File): Promise<XrayPredictionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await apiClient.post<XrayPredictionResponse>(
    "/predict/xray",
    formData,
    { headers: { "Content-Type": "multipart/form-data" } }
  );
  return response.data;
}

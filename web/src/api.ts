import type { Job, Predictor, Recovery, Rect, Share } from "./types";

const API = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

async function json<T>(response: Response): Promise<T> {
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail ?? "Request failed");
  return body as T;
}

export async function encodeOwner(file: File, seed: number, predictor: Predictor, rect: Rect, msaMode: Job["msaMode"]): Promise<Job> {
  const form = new FormData();
  form.append("file", file);
  form.append("seed", String(seed));
  form.append("predictor", predictor);
  form.append("top", String(rect.top));
  form.append("left", String(rect.left));
  form.append("height", String(rect.height));
  form.append("width", String(rect.width));
  form.append("msa_mode", msaMode);
  return json<Job>(await fetch(`${API}/api/owner/encode`, { method: "POST", body: form }));
}

export async function recover(jobId: string, stage: number, shareIndexes: number[]): Promise<Recovery> {
  return json<Recovery>(await fetch(`${API}/api/receiver/recover`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ jobId, stage, shareIndexes }),
  }));
}

export async function recoverAutomatically(jobId: string, stage: number): Promise<Recovery> {
  return json<Recovery>(await fetch(`${API}/api/receiver/recover-auto`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ jobId, stage, shareIndexes: [] }),
  }));
}

export async function embedPayload(jobId: string, shareIndex: number, payload: string): Promise<{ message: string; share: Share; crc: string }> {
  return json(await fetch(`${API}/api/data-hider/embed`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ jobId, shareIndex, payload }),
  }));
}

export async function extractPayload(jobId: string, shareIndex: number): Promise<{ payload: string; crc: string }> {
  return json(await fetch(`${API}/api/data-hider/extract`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ jobId, shareIndex }),
  }));
}

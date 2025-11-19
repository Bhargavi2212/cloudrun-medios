import axios from "axios";

const createClient = (envKey: string, fallbackPort: number) => {
  const baseURL = import.meta.env[envKey] ?? `http://localhost:${fallbackPort}`;
  return axios.create({
    baseURL,
    headers: {
      "Content-Type": "application/json",
    },
  });
};

export const manageApi = createClient("VITE_MANAGE_API_URL", 9001);
export const scribeApi = createClient("VITE_SCRIBE_API_URL", 9002);
export const summarizerApi = createClient("VITE_SUMMARIZER_API_URL", 9003);
export const dolApi = createClient("VITE_DOL_API_URL", 8004);
export const federationApi = createClient("VITE_FEDERATION_API_URL", 8010);


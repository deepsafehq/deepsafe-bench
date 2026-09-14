import { API_URL } from "@/lib/config";

export { API_URL };

/** Shared fetch wrapper — returns parsed JSON or throws with a human-readable message. */
async function request<T>(
  path: string,
  options: RequestInit & { token?: string } = {},
): Promise<T> {
  const { token, ...fetchOptions } = options;
  const headers: Record<string, string> = {
    ...(fetchOptions.headers as Record<string, string>),
  };

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_URL}${path}`, { ...fetchOptions, headers });

  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    try {
      const data = await res.json();
      const detail = data?.detail;
      if (typeof detail === "string") message = detail;
      else if (detail?.message) message = detail.message;
    } catch {
      // ignore parse errors
    }
    throw new Error(message);
  }

  return res.json() as Promise<T>;
}

/** Reusable API client with shared fetch, auth, and error handling. */
export const apiClient = {
  get<T>(path: string, token?: string): Promise<T> {
    return request<T>(path, { method: "GET", token });
  },

  post<T>(path: string, body: unknown, token?: string): Promise<T> {
    return request<T>(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      token,
    });
  },

  delete<T>(path: string, token?: string): Promise<T> {
    return request<T>(path, { method: "DELETE", token });
  },
};

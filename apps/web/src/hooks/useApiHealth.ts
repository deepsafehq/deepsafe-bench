import { useState, useEffect, useCallback } from 'react';
import { API_URL } from '@/lib/api';

function useApiHealth(intervalMs = 30000): { apiOnline: boolean; checking: boolean } {
  const [apiOnline, setApiOnline] = useState(false);
  const [checking, setChecking] = useState(true);

  const checkHealth = useCallback(async () => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      const res = await fetch(`${API_URL}/health`, { signal: controller.signal });
      clearTimeout(timeoutId);
      setApiOnline(res.ok);
    } catch {
      setApiOnline(false);
    } finally {
      setChecking(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, intervalMs);
    return () => clearInterval(interval);
  }, [checkHealth, intervalMs]);

  return { apiOnline, checking };
}

export { useApiHealth };

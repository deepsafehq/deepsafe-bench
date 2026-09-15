"use client";

import { useCallback, useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AppNav } from "@/components/app-nav";
import { Key, Plus, Trash2, Copy, Check } from "lucide-react";
import { useAuth } from "@/components/auth-provider";
import { AuthDisabledNotice } from "@/components/auth-disabled-notice";

import { API_URL } from "@/lib/api";

interface ApiKeyRow {
  id: string;
  key_prefix: string;
  name: string;
  tier: string;
  scans_used: number;
  created_at: string;
  last_used_at: string | null;
}

export default function ApiKeysPage() {
  const { session, isLoading: authLoading, user , supabase } = useAuth();
  // No Supabase client means auth is not configured for this build.
  const authConfigured = Boolean(supabase);
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [keys, setKeys] = useState<ApiKeyRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [newKeyName, setNewKeyName] = useState("");
  const [showCreate, setShowCreate] = useState(false);
  const [createdKey, setCreatedKey] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // AuthProvider handles redirect for unauthenticated users.
  // Extract access token from session once auth is resolved.
  useEffect(() => {
    if (authLoading || !user) return;
    if (session?.access_token) setAccessToken(session.access_token);
  }, [authLoading, user, session]);

  const fetchKeys = useCallback(async () => {
    if (!accessToken) return;
    try {
      const res = await fetch(`${API_URL}/api/keys`, {
        headers: { Authorization: `Bearer ${accessToken}` },
      });
      if (res.ok) setKeys(await res.json());
    } catch (e) {
      console.error("Failed to fetch keys", e);
    } finally {
      setLoading(false);
    }
  }, [accessToken]);

  useEffect(() => {
    fetchKeys();
  }, [fetchKeys]);

  const MAX_KEYS = 5;
  const [error, setError] = useState<string | null>(null);

  const createKey = async () => {
    if (!accessToken || !newKeyName.trim()) return;
    if (keys.length >= MAX_KEYS) return;
    setError(null);
    try {
      const res = await fetch(`${API_URL}/api/keys`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ name: newKeyName.trim() }),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        setError(errData?.detail || `Failed to create key (${res.status})`);
        return;
      }
      const data = await res.json();
      setCreatedKey(data.key);
      setNewKeyName("");
      fetchKeys();
    } catch {
      setError("Network error. Please check your connection and try again.");
    }
  };

  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);

  const revokeKey = async (id: string) => {
    if (!accessToken) return;
    setConfirmDeleteId(null);
    setKeys((prev) => prev.filter((k) => k.id !== id));
    setError(null);
    try {
      const res = await fetch(`${API_URL}/api/keys/${id}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${accessToken}` },
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        setError(errData?.detail || `Failed to revoke key (${res.status})`);
        fetchKeys();
      }
    } catch {
      setError("Network error. Please check your connection and try again.");
      fetchKeys();
    }
  };

  const copyKey = (key: string) => {
    navigator.clipboard.writeText(key);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const curlSnippet = createdKey
    ? `curl -X POST http://localhost:8000/v1/detect \\
  -H "Authorization: Bearer ${createdKey}" \\
  -F "file=@photo.jpg"`
    : "";

  if (!authConfigured) {
    return <AuthDisabledNotice page="API key management" />;
  }

  return (
    <div
      data-theme="app"
      className="min-h-screen bg-background text-text-primary"
    >
      <AppNav />
      <div className="max-w-4xl mx-auto px-4 sm:px-6 py-8">
        {error && (
          <div
            role="alert"
            className="mb-4 px-4 py-3 rounded-md bg-danger-light border border-danger/20 text-sm text-danger"
          >
            {error}
          </div>
        )}
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-2xl font-heading font-light text-text-primary">
            API Keys
          </h1>
          <button
            onClick={() => {
              setShowCreate(true);
              setCreatedKey(null);
            }}
            disabled={keys.length >= MAX_KEYS}
            aria-label="Create API key"
            className="flex items-center gap-2 px-4 py-2 bg-accent text-accent-foreground rounded-md text-sm font-medium hover:bg-accent-hover transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <Plus className="w-4 h-4" /> Create Key
          </button>
        </div>

        {/* Create key dialog */}
        {showCreate && (
          <div className="border border-border rounded-md p-6 mb-6 bg-surface">
            {createdKey ? (
              <div>
                <p className="text-sm font-medium mb-2">
                  Your new API key (copy it now &mdash; you won&apos;t see it
                  again):
                </p>
                <div className="flex items-center gap-2 bg-accent-light border border-accent/20 rounded-md p-3 font-mono text-sm">
                  <code className="flex-1 break-all">{createdKey}</code>
                  <button
                    onClick={() => copyKey(createdKey)}
                    aria-label={copied ? "Copied" : "Copy API key"}
                    className="shrink-0"
                  >
                    {copied ? (
                      <Check className="w-4 h-4 text-accent" />
                    ) : (
                      <Copy className="w-4 h-4 text-accent hover:text-accent-hover" />
                    )}
                  </button>
                </div>
                <div className="mt-4">
                  <p className="text-sm font-medium mb-2">Quick start:</p>
                  <pre className="bg-background border border-border rounded-md p-3 text-xs overflow-x-auto">
                    {curlSnippet}
                  </pre>
                </div>
                <button
                  onClick={() => {
                    setShowCreate(false);
                    setCreatedKey(null);
                  }}
                  className="mt-4 px-4 py-2 text-sm text-text-secondary border border-border rounded-md hover:bg-surface hover:text-text-primary transition-colors"
                >
                  Done
                </button>
              </div>
            ) : (
              <div className="flex items-end gap-3">
                <div className="flex-1">
                  <label
                    htmlFor="new-key-name"
                    className="text-sm font-medium mb-1 block"
                  >
                    Key Name
                  </label>
                  <input
                    id="new-key-name"
                    type="text"
                    placeholder="e.g., Production, Testing"
                    value={newKeyName}
                    onChange={(e) => setNewKeyName(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && createKey()}
                    className="w-full px-3 py-2 bg-background border border-border rounded-md text-sm text-text-primary"
                  />
                </div>
                <button
                  onClick={createKey}
                  disabled={!newKeyName.trim()}
                  className="px-4 py-2 bg-accent text-accent-foreground rounded-md text-sm font-medium hover:bg-accent-hover disabled:opacity-50"
                >
                  Create
                </button>
                <button
                  onClick={() => setShowCreate(false)}
                  className="px-4 py-2 text-sm text-text-secondary hover:text-text-primary"
                >
                  Cancel
                </button>
              </div>
            )}
          </div>
        )}

        {/* Keys list */}
        {loading ? (
          <p className="text-sm text-text-secondary">Loading...</p>
        ) : keys.length === 0 ? (
          <div className="border border-dashed border-border rounded-md p-12 text-center">
            <Key className="w-8 h-8 mx-auto text-text-secondary mb-3" />
            <p className="text-sm text-text-secondary">
              No API keys yet. Create one to get started.
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            <AnimatePresence mode="popLayout">
              {keys.map((k) => (
                <motion.div
                  key={k.id}
                  layout
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, x: -40, scale: 0.95 }}
                  transition={{ duration: 0.25, ease: "easeOut" }}
                  className="border border-border rounded-md p-4 bg-surface flex items-center justify-between"
                >
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-sm">{k.name}</span>
                      <span className="bg-accent-light text-accent rounded-full text-xs px-2 py-0.5 uppercase">
                        {k.tier}
                      </span>
                    </div>
                    <div className="flex items-center gap-4 mt-1 text-xs text-text-tertiary">
                      <span className="font-mono text-sm text-text-secondary">
                        {k.key_prefix}...
                      </span>
                      <span>{k.scans_used} scans</span>
                      <span>
                        Created {new Date(k.created_at).toLocaleDateString()}
                      </span>
                      {k.last_used_at && (
                        <span>
                          Last used{" "}
                          {new Date(k.last_used_at).toLocaleDateString()}
                        </span>
                      )}
                    </div>
                  </div>
                  {confirmDeleteId === k.id ? (
                    <div className="flex items-center gap-2 shrink-0">
                      <span className="text-xs text-text-secondary">
                        Delete?
                      </span>
                      <button
                        onClick={() => revokeKey(k.id)}
                        className="px-2 py-1 text-xs font-medium text-danger border border-danger/30 rounded-sm hover:bg-danger-light transition-colors"
                      >
                        Yes
                      </button>
                      <button
                        onClick={() => setConfirmDeleteId(null)}
                        className="px-2 py-1 text-xs font-medium text-text-secondary border border-border rounded-sm hover:bg-surface transition-colors"
                      >
                        No
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => setConfirmDeleteId(k.id)}
                      className="text-text-secondary hover:text-danger hover:bg-danger-light p-2 rounded-sm transition-colors"
                      aria-label={`Revoke API key ${k.name}`}
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  )}
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>
    </div>
  );
}

"use client";

import {
  createContext,
  useContext,
  useEffect,
  useState,
  useCallback,
} from "react";
import { useRouter, usePathname } from "next/navigation";
import { createClient } from "@/lib/supabase";
import type { User, Session } from "@supabase/supabase-js";

/**
 * Pages that never require authentication.
 *
 * Auth is opt-in. With NEXT_PUBLIC_AUTH_ENABLED unset, every page is public,
 * which is the correct default for a static build with no gateway behind it.
 */
const PUBLIC_PATHS = ["/login", "/auth/callback", "/privacy", "/terms"];

interface AuthContextValue {
  user: User | null;
  session: Session | null;
  supabase: ReturnType<typeof createClient>;
  isLoading: boolean;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue>({
  user: null,
  session: null,
  supabase: null,
  isLoading: true,
  signOut: async () => {},
});

export function useAuth() {
  return useContext(AuthContext);
}

/**
 * Centralized auth provider. Manages Supabase auth state for the entire app.
 *
 * - Listens to onAuthStateChange for reactive session tracking.
 * - Gates rendering: shows a blank dark screen while auth state resolves,
 *   preventing flash of login form (when authenticated) or flash of
 *   protected content (when unauthenticated).
 * - Handles auto-redirect: authenticated users visiting /login get sent to /,
 *   unauthenticated users visiting protected pages get sent to /login.
 *
 * API key management is NOT handled here — the web UI passes the Supabase
 * JWT directly to the /v1/ endpoints.  API keys are only relevant to
 * external developers and are managed on the /api-keys page.
 */
export function AuthProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const [supabase] = useState(() => createClient());
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const authEnabled = process.env.NEXT_PUBLIC_AUTH_ENABLED === "true";

  useEffect(() => {
    // No client means auth is not configured; resolve immediately as signed out.
    if (!supabase) {
      setIsLoading(false);
      return;
    }

    supabase.auth
      .getSession()
      .then(({ data }: { data: { session: Session | null } }) => {
        setSession(data.session);
        setUser(data.session?.user ?? null);
        setIsLoading(false);
      });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange(
      (_event: string, newSession: Session | null) => {
        setSession(newSession);
        setUser(newSession?.user ?? null);
        setIsLoading(false);
      },
    );

    return () => {
      subscription.unsubscribe();
    };
  }, [supabase]);

  // Handle redirects after auth state is resolved.
  // Only enforce auth when a gateway is actually configured.
  useEffect(() => {
    if (isLoading) return;

    const isPublicPath =
      pathname === "/" ||
      PUBLIC_PATHS.some((p) => pathname.startsWith(p));

    if (user && pathname === "/login") {
      window.location.href = `${window.location.origin}/`;
    } else if (!user && !isPublicPath && authEnabled) {
      router.replace("/login");
    }
  }, [isLoading, user, pathname, router, authEnabled]);

  const signOut = useCallback(async () => {
    localStorage.removeItem("deepsafe_api_key");
    await supabase?.auth.signOut();
    window.location.href = `${window.location.origin}/`;
  }, [supabase]);

  // Gate rendering behind auth resolution only on the app domain.
  // The landing site (deepsafehq.github.io/deepsafe-bench) is fully public and should render immediately.
  if (isLoading && authEnabled) {
    return (
      <AuthContext.Provider
        value={{ user, session, supabase, isLoading, signOut }}
      >
        <div data-theme="app" className="min-h-screen bg-background" />
      </AuthContext.Provider>
    );
  }

  return (
    <AuthContext.Provider
      value={{ user, session, supabase, isLoading, signOut }}
    >
      {children}
    </AuthContext.Provider>
  );
}

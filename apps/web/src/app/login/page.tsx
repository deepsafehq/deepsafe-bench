"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  Mail,
  Lock,
  ArrowRight,
  AlertCircle,
  CheckCircle2,
} from "lucide-react";
import { DeepSafeLogo } from "@/components/logo";
import { useAuth } from "@/components/auth-provider";

export default function LoginPage() {
  const { user, supabase, isLoading: authLoading } = useAuth();
  const [isLoginMode, setIsLoginMode] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [signupSuccess, setSignupSuccess] = useState(false);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("error") === "auth_failed") {
      setErrorMsg("Sign-in failed. Please try again.");
    }
  }, []);

  // AuthProvider handles redirect for authenticated users on /login.
  // If still loading or user is set, don't render the form.
  if (authLoading || user) {
    return <div data-theme="app" className="min-h-screen bg-background" />;
  }

  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setErrorMsg(null);
    setSignupSuccess(false);

    try {
      if (isLoginMode) {
        const { error } = await supabase.auth.signInWithPassword({
          email,
          password,
        });
        if (error) {
          throw new Error(error.message);
        }
        window.location.href = `${window.location.origin}/`;
      } else {
        const { error } = await supabase.auth.signUp({
          email,
          password,
        });
        if (error) {
          throw new Error(error.message);
        }
        setSignupSuccess(true);
        setIsLoading(false);
      }
    } catch (err: unknown) {
      const message =
        err instanceof Error
          ? err.message
          : "Authentication failed. Please try again.";
      setErrorMsg(message);
      setIsLoading(false);
    }
  };

  const handleGoogleSignIn = async () => {
    const { error } = await supabase.auth.signInWithOAuth({
      provider: "google",
      options: {
        redirectTo: `${window.location.origin}/auth/callback`,
      },
    });
    if (error) {
      setErrorMsg(error.message);
    }
  };

  return (
    <div
      data-theme="app"
      className="min-h-screen bg-background text-text-primary font-sans selection:bg-accent/30 flex flex-col items-center justify-center relative overflow-hidden"
    >
      <div className="relative z-10 w-full max-w-md px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="bg-surface border border-border rounded-md p-8 shadow-md relative overflow-hidden"
        >
          <div className="absolute top-0 left-0 right-0 h-px bg-accent" />

          <div className="flex flex-col items-center mb-8">
            <Link href="/" className="flex items-center mb-6">
              <DeepSafeLogo size="md" variant="dark" />
            </Link>
            <h1 className="text-2xl font-heading font-light text-text-primary mb-2">
              {isLoginMode ? "Welcome back" : "Create your account"}
            </h1>
            <p className="text-sm text-text-secondary text-center">
              {isLoginMode
                ? "Sign in to access your Dashboard"
                : "Sign up to access the Dashboard"}
            </p>
          </div>

          {errorMsg && (
            <div
              role="alert"
              className="mb-6 p-3 bg-danger-light border border-danger/20 rounded-md flex items-start gap-3"
            >
              <AlertCircle className="w-5 h-5 text-danger shrink-0 mt-0.5" />
              <p className="text-sm text-danger">{errorMsg}</p>
            </div>
          )}

          {signupSuccess && (
            <div
              role="alert"
              className="mb-6 p-3 bg-accent-light border border-accent/20 rounded-md flex items-start gap-3"
            >
              <CheckCircle2 className="w-5 h-5 text-accent shrink-0 mt-0.5" />
              <p className="text-sm text-accent">
                Check your email to verify your account.
              </p>
            </div>
          )}

          <div className="space-y-4">
            <button
              type="button"
              onClick={handleGoogleSignIn}
              aria-label="Continue with Google"
              className="w-full h-12 bg-surface-raised border border-border text-text-primary font-medium rounded-md flex items-center justify-center gap-3 hover:bg-surface transition-colors"
            >
              <svg
                viewBox="0 0 24 24"
                width="20"
                height="20"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                  fill="#4285F4"
                />
                <path
                  d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                  fill="#34A853"
                />
                <path
                  d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                  fill="#FBBC05"
                />
                <path
                  d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                  fill="#EA4335"
                />
              </svg>
              Continue with Google
            </button>
          </div>

          <div className="my-6 flex items-center gap-4">
            <div className="h-px bg-border flex-1" />
            <span className="text-xs text-text-tertiary uppercase tracking-wider">
              Or continue with email
            </span>
            <div className="h-px bg-border flex-1" />
          </div>

          <form onSubmit={handleAuth} className="space-y-4">
            <div>
              <label
                htmlFor="login-email"
                className="block text-sm font-medium text-text-secondary mb-1.5"
              >
                Email
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-text-tertiary" />
                <input
                  id="login-email"
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  className="w-full h-12 bg-background border border-border rounded-md pl-10 pr-4 text-text-primary placeholder:text-text-tertiary focus:outline-none focus:border-accent/50 focus:ring-1 focus:ring-accent/50 transition-all"
                />
              </div>
            </div>
            <div>
              <label
                htmlFor="login-password"
                className="block text-sm font-medium text-text-secondary mb-1.5"
              >
                Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-text-tertiary" />
                <input
                  id="login-password"
                  type="password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full h-12 bg-background border border-border rounded-md pl-10 pr-4 text-text-primary placeholder:text-text-tertiary focus:outline-none focus:border-accent/50 focus:ring-1 focus:ring-accent/50 transition-all"
                />
              </div>
            </div>
            <button
              type="submit"
              disabled={isLoading}
              className="w-full h-12 bg-accent text-accent-foreground font-medium rounded-md flex items-center justify-center gap-2 hover:bg-accent-hover transition-colors mt-2"
            >
              {isLoading ? (
                <div className="w-5 h-5 border-2 border-black/20 border-t-black rounded-full animate-spin" />
              ) : (
                <>
                  {isLoginMode ? "Log In" : "Sign Up"}{" "}
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </form>

          <p className="mt-8 text-center text-sm text-text-tertiary">
            {isLoginMode
              ? "Don't have an account? "
              : "Already have an account? "}
            <button
              type="button"
              onClick={() => setIsLoginMode(!isLoginMode)}
              className="text-accent hover:text-accent-hover"
            >
              {isLoginMode ? "Sign up" : "Log in"}
            </button>
          </p>
        </motion.div>
      </div>
    </div>
  );
}

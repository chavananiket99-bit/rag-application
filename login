import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  AlertCircle,
  ArrowRight,
  Eye,
  EyeOff,
  LockKeyhole,
  LogIn,
  ShieldCheck,
  UserRound,
} from "lucide-react";
import { useAuth } from "../context/AuthContext";
export default function Login() {
  const navigate = useNavigate();
  const location = useLocation();
  const { login, loginWithSSO } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [ssoLoading, setSsoLoading] = useState(false);
  const [error, setError] = useState("");
  const redirectPath = location.state?.from || "/dashboard";

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (loading || ssoLoading) {
      return;
    }
    setError("");
    setLoading(true);
    try {
      await login({
        username,
        password,
      });
      navigate(redirectPath, {
        replace: true,
      });
    } catch (err) {
      setError(
        err?.message || "Unable to sign in. Please check your credentials.",
      );
    } finally {
      setLoading(false);
    }
  };

  const handleSSO = async () => {
    if (loading || ssoLoading) {
      return;
    }
    setError("");
    setSsoLoading(true);

    try {
      await loginWithSSO();
    } catch (err) {
      setError(err?.message || "SSO sign-in is currently unavailable.");
    } finally {
      setSsoLoading(false);
    }
  };
  return (
    <div className="min-h-dvh bg-slate-950 tracking-[0.015em] lg:h-dvh lg:overflow-hidden">
      <div className="grid min-h-dvh lg:h-full lg:min-h-0 lg:grid-cols-[1.1fr_0.9fr]">
        {/* Left branding panel */}
        <div className="relative hidden overflow-hidden lg:flex">
          <div className="absolute inset-0 bg-slate-950" />
          <div className="absolute -left-32 top-20 h-96 w-96 rounded-full bg-red-700/20 blur-3xl" />
          <div className="absolute bottom-0 right-0 h-96 w-96 rounded-full bg-red-700/10 blur-3xl" />
          <div className="relative flex w-full flex-col justify-between p-10 xl:p-12">
            <div className="flex items-center gap-4">
              <div className="flex items-center justify-center">
                <img
                  src="/mahindra-logo.png"
                  alt="Mahindra"
                  className="object-contain w-[200px]"
                />
              </div>
              {/* <div>
                <p className="text-sm font-semibold text-white">AI</p>
                <p className="text-xs text-slate-500">Knowledge Portal</p>
              </div> */}
            </div>
            <div className="max-w-xl">
              <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-slate-800 bg-slate-900/80 px-3 py-1.5">
                <ShieldCheck size={14} className="text-emerald-400" />
                <span className="text-xs font-medium text-slate-300">
                  Enterprise Knowledge Workspace
                </span>
              </div>
              <h1 className="text-4xl font-semibold tracking-tight text-white xl:text-5xl">
                Your enterprise knowledge,
                <span className="text-red-500"> with AI.</span>
              </h1>
              <p className="mt-5 max-w-lg text-sm leading-7 text-slate-400">
                Upload enterprise documents, ask questions in natural language,
                and get AI-assisted answers grounded in your organization's
                knowledge.
              </p>
            </div>
            <p className="text-xs text-slate-600">
              Secure enterprise AI workspace
            </p>
          </div>
        </div>
        {/* Login panel */}
        <div className="flex items-center justify-center bg-slate-100 px-5 py-5 sm:px-8 lg:h-full lg:min-h-0">
          <div className="w-full max-w-md">
            {/* Mobile branding */}
            <div className="mb-8 flex items-center gap-3 lg:hidden">
              <div className="flex items-center justify-center">
                <img
                  src="/mahindra-logo.png"
                  alt="Mahindra"
                  className="object-contain w-[150px]"
                />
              </div>
              {/* <div>
                <p className="text-sm font-semibold text-slate-800">
                  AI
                </p>
                <p className="text-xs text-slate-500">Knowledge Portal</p>
              </div> */}
            </div>
            <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-xl sm:p-7">
              <div className="mb-5">
                <h2 className="text-2xl font-semibold tracking-tight text-slate-900">
                  Welcome back
                </h2>
                <p className="mt-2 text-sm leading-6 text-slate-500">
                  Sign in to access your AI knowledge workspace.
                </p>
              </div>
              {error && (
                <div className="mb-5 flex items-start gap-2 rounded-xl border border-red-200 bg-red-50 px-3.5 py-3 text-sm text-red-700">
                  <AlertCircle size={17} className="mt-0.5 shrink-0" />
                  <span>{error}</span>
                </div>
              )}
              {/* SSO */}
              <button
                type="button"
                onClick={handleSSO}
                disabled={loading || ssoLoading}
                className="flex w-full items-center justify-center gap-2 rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm font-semibold text-slate-700 shadow-sm transition hover:border-slate-400 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {ssoLoading ? (
                  <>
                    <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-slate-700" />
                    Connecting to SSO...
                  </>
                ) : (
                  <>
                    <ShieldCheck size={18} />
                    Continue with Corporate SSO
                  </>
                )}
              </button>
              {/* Divider */}
              <div className="my-4 flex items-center gap-3">
                <div className="h-px flex-1 bg-slate-200" />
                <span className="text-[11px] font-medium uppercase tracking-wider text-slate-400">
                  or sign in with account
                </span>
                <div className="h-px flex-1 bg-slate-200" />
              </div>
              <form onSubmit={handleSubmit} className="space-y-3">
                {/* Username */}
                <div>
                  <label
                    htmlFor="username"
                    className="mb-1.5 block text-xs font-semibold text-slate-700"
                  >
                    Username
                  </label>
                  <div className="relative">
                    <UserRound
                      size={17}
                      className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"
                    />
                    <input
                      id="username"
                      type="text"
                      autoComplete="username"
                      value={username}
                      onChange={(event) => setUsername(event.target.value)}
                      placeholder="Enter your username"
                      disabled={loading || ssoLoading}
                      className="w-full rounded-xl border border-slate-300 bg-white py-3 pl-10 pr-3 text-sm text-slate-700 outline-none transition placeholder:text-slate-400 focus:border-red-400 focus:ring-2 focus:ring-red-100 disabled:bg-slate-50"
                    />
                  </div>
                </div>
                {/* Password */}
                <div>
                  <label
                    htmlFor="password"
                    className="mb-1.5 block text-xs font-semibold text-slate-700"
                  >
                    Password
                  </label>
                  <div className="relative">
                    <LockKeyhole
                      size={17}
                      className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"
                    />
                    <input
                      id="password"
                      type={showPassword ? "text" : "password"}
                      autoComplete="current-password"
                      value={password}
                      onChange={(event) => setPassword(event.target.value)}
                      placeholder="Enter your password"
                      disabled={loading || ssoLoading}
                      className="w-full rounded-xl border border-slate-300 bg-white py-3 pl-10 pr-11 text-sm text-slate-700 outline-none transition placeholder:text-slate-400 focus:border-red-400 focus:ring-2 focus:ring-red-100 disabled:bg-slate-50"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword((value) => !value)}
                      className="absolute right-2 top-1/2 -translate-y-1/2 rounded-lg p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
                      aria-label={
                        showPassword ? "Hide password" : "Show password"
                      }
                    >
                      {showPassword ? <EyeOff size={17} /> : <Eye size={17} />}
                    </button>
                  </div>
                </div>
                <button
                  type="submit"
                  disabled={
                    loading || ssoLoading || !username.trim() || !password
                  }
                  className="mt-2 flex w-full items-center justify-center gap-2 rounded-xl bg-red-700 px-4 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-red-800 disabled:cursor-not-allowed disabled:bg-slate-300"
                >
                  {loading ? (
                    <>
                      <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" />
                      Signing in...
                    </>
                  ) : (
                    <>
                      <LogIn size={18} />
                      Sign in
                      <ArrowRight size={17} />
                    </>
                  )}
                </button>
              </form>
              <div className="mt-6 rounded-lg bg-slate-50 p-4">
                <p className="mb-3 text-sm font-semibold text-slate-700">
                  Development Logins
                </p>
                <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-sm">
                  <span className="whitespace-nowrap">
                    <span className="font-medium">demo</span>
                    <span className="text-slate-500"> / demo123</span>
                  </span>
                  <span className="whitespace-nowrap">
                    <span className="font-medium">user1</span>
                    <span className="text-slate-500"> / user1@123</span>
                  </span>
                  <span className="whitespace-nowrap">
                    <span className="font-medium">user2</span>
                    <span className="text-slate-500"> / user2@123</span>
                  </span>
                </div>
              </div>
            </div>
            <p className="mt-3 text-center text-[11px] text-slate-400">
              Authorized users only
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

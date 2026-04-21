/**
 * KrishiMitra – API Client v5
 * KEY FIX: runAnalysis now sends state + district as SEPARATE fields
 * matching the backend AnalysisRequest schema exactly.
 */
const API = (() => {
  const BASE_URL = window.location.hostname === "localhost"
    ? "http://localhost:8000"
    : "https://agri-backend-pjw5.onrender.com";

  function getToken()     { return localStorage.getItem("km_token"); }
  function setToken(t)    { localStorage.setItem("km_token", t); }
  function getUser()      { return JSON.parse(localStorage.getItem("km_user") || "null"); }
  function setUser(u)     { localStorage.setItem("km_user", JSON.stringify(u)); }
  function clearSession() { localStorage.removeItem("km_token"); localStorage.removeItem("km_user"); }

  function isLoggedIn() {
    const t = getToken();
    if (!t) return false;
    try { const p = JSON.parse(atob(t.split(".")[1])); return p.exp * 1000 > Date.now(); }
    catch { return false; }
  }

  function resolveRoot(path) {
    return window.location.pathname.includes("/pages/") ? "../" + path : path;
  }

  async function request(path, options = {}) {
    const token = getToken();
    const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
    if (token) headers["Authorization"] = `Bearer ${token}`;
    const res  = await fetch(BASE_URL + path, { ...options, headers });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Request failed");
    return data;
  }

  // Auth
  async function sendOTP(email, name) {
    return request("/auth/send-otp", { method:"POST", body: JSON.stringify({email, name}) });
  }
  async function verifyOTP(email, otp) {
    const data = await request("/auth/verify-otp", { method:"POST", body: JSON.stringify({email, otp}) });
    if (data.token) { setToken(data.token); setUser(data.user); }
    return data;
  }
  async function getMe() { return request("/auth/me"); }
  function logout() { clearSession(); window.location.href = resolveRoot("pages/login.html"); }

  /**
   * KEY FIX: state and district sent as SEPARATE fields.
   * Backend AnalysisRequest expects: { state, district, crop, season }
   * NOT: { location, crop, season }
   * 
   * @param {string} state    - e.g. "Kerala"
   * @param {string} district - e.g. "Thrissur" (optional, can be "")
   * @param {string} crop     - e.g. "Rice"
   * @param {string} season   - e.g. "Kharif" (optional)
   */
  async function runAnalysis(state, district, crop, season) {
    return request("/analysis/run", {
      method: "POST",
      body: JSON.stringify({ state, district, crop, season })
    });
  }

  async function getHistory()   { return request("/analysis/history"); }
  async function getAnalysis(id){ return request(`/analysis/${id}`); }
  async function submitContact(name, email, subject, message) {
    return request("/contact", { method:"POST", body: JSON.stringify({name,email,subject,message}) });
  }
  async function getCrops(search="") {
    const q = search ? `?search=${encodeURIComponent(search)}` : "";
    return request(`/crops${q}`);
  }
  async function getMLInfo()        { return request("/ml/info"); }
  async function getStateData(state){ return request(`/ml/state/${encodeURIComponent(state)}`); }

  function requireAuth() {
    if (!isLoggedIn()) window.location.href = resolveRoot("pages/login.html");
  }

  return {
    sendOTP, verifyOTP, getMe, logout,
    runAnalysis, getHistory, getAnalysis,
    submitContact, getCrops, getMLInfo, getStateData,
    getToken, getUser, isLoggedIn, requireAuth,
    resolveRoot, BASE_URL
  };
})();
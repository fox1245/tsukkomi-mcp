// Opt-in OMP user extension. Register this file in OMP's user extensions and set
// TSUKKOMI_OMP_MONITOR=1, SELF_DIRECT_OMP_SESSIONS_DIR and SELF_DIRECT_INDEX_DIR.
// TSUKKOMI_OMP_PYTHON selects the Python interpreter with tsukkomi-mcp installed;
// TSUKKOMI_OMP_SOURCE can point at a source checkout without changing OMP's PYTHONPATH.
// SELF_DIRECT_CONTRACTS_PATH may select an existing private contracts file.
// Coverage: only OMP hook events. Pre-tool checks can block confirmed findings;
// tool_result and turn_end re-audit the currently persisted JSONL snapshot (a
// hook may precede disk flush). In-memory sessions, skipped hooks, and transport
// failures are NOT monitored or evidence of compliance.
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { delimiter } from "node:path";
import { StringDecoder } from "node:string_decoder";

type Context = {
  sessionManager: { getSessionId(): string; getSessionFile(): string | undefined };
  ui?: { notify(message: string, level: "info" | "warning" | "error"): void };
};
type HookAPI = { on(event: string, handler: (event: Record<string, unknown>, ctx: Context) => unknown): void };
type Response = { id?: number; result?: unknown; error?: unknown };
type Audit = { ok: true; verdict: "violation" | "suspicious" | "clean" | "unknown"; applicable_contracts?: number; findings?: Array<{ contract_id?: string; reason?: string; verdict?: string }> };
type Session = { session_id: string; provider: "omp"; path: string };

const enabled = process.env.TSUKKOMI_OMP_MONITOR === "1";
const root = process.env.SELF_DIRECT_OMP_SESSIONS_DIR;
const maxLine = 4 * 1024 * 1024;
const maxRequest = 256 * 1024;
const timeoutMs = 12000;
const actionable: Record<string, true> = { violation: true, suspicious: true };

class LocalMcp {
  private child?: ChildProcessWithoutNullStreams;
  private buffer = "";
  private decoder = new StringDecoder("utf8");
  private nextId = 0;
  private pending = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void; timer: ReturnType<typeof setTimeout> }>();
  private starting?: Promise<void>;

  private stop(reason: string): void {
    const child = this.child;
    this.child = undefined;
    this.starting = undefined;
    this.buffer = "";
    this.decoder = new StringDecoder("utf8");
    for (const item of this.pending.values()) {
      clearTimeout(item.timer);
      item.reject(new Error(reason));
    }
    this.pending.clear();
    child?.kill();
  }

  close(): void { this.stop("bridge closed"); }

  private send(method: string, params: object): Promise<unknown> {
    const child = this.child;
    if (!child || !child.stdin.writable) return Promise.reject(new Error("bridge unavailable"));
    const id = ++this.nextId;
    const line = JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n";
    if (Buffer.byteLength(line) > maxRequest) return Promise.reject(new Error("request too large to monitor"));
    const { promise, resolve, reject } = Promise.withResolvers<unknown>();
    const timer = setTimeout(() => this.stop("bridge timed out"), timeoutMs);
    this.pending.set(id, { resolve, reject, timer });
    try { child.stdin.write(line, (err) => { if (err) this.stop("bridge write failed"); }); }
    catch { this.stop("bridge write failed"); }
    return promise;
  }

  private async start(): Promise<void> {
    if (this.starting) return this.starting;
    if (this.child) return;
    this.starting = (async () => {
      if (!root || !process.env.SELF_DIRECT_INDEX_DIR) throw new Error("private runtime locations not configured");
      const env = { ...process.env, SELF_DIRECT_LOCAL_ONLY: "true", SELF_DIRECT_OMP_SESSIONS_DIR: root };
      const source = process.env.TSUKKOMI_OMP_SOURCE;
      if (source) env.PYTHONPATH = env.PYTHONPATH ? `${source}${delimiter}${env.PYTHONPATH}` : source;
      // The local-only engine must never select an inherited remote credential.
      delete env.OPENROUTER_API_KEY;
      delete env.SELF_DIRECT_OPENROUTER_API_KEY_FILE;
      const child = spawn(process.env.TSUKKOMI_OMP_PYTHON || "python", ["-m", "self_directing_mcp.server"],
        { env, stdio: ["pipe", "pipe", "pipe"], windowsHide: true });
      this.child = child;
      child.stderr.on("data", () => { /* drain without logging potentially sensitive diagnostics */ });
      child.on("error", () => this.stop("bridge process failed"));
      child.on("exit", () => { if (this.child === child) this.stop("bridge process exited"); });
      child.stdout.on("data", (bytes: Buffer) => {
        if (this.child !== child) return;
        this.buffer += this.decoder.write(bytes);
        if (Buffer.byteLength(this.buffer) > maxLine) { this.stop("bridge response too large"); return; }
        let end: number;
        while ((end = this.buffer.indexOf("\n")) !== -1) {
          const line = this.buffer.slice(0, end);
          this.buffer = this.buffer.slice(end + 1);
          let response: Response;
          try { response = JSON.parse(line); } catch { this.stop("invalid bridge response"); return; }
          if (typeof response.id !== "number") continue;
          const item = this.pending.get(response.id);
          if (!item) continue;
          this.pending.delete(response.id);
          clearTimeout(item.timer);
          if (response.error) item.reject(new Error("MCP error"));
          else item.resolve(response.result);
        }
      });
      await this.send("initialize", { protocolVersion: "2025-03-26", capabilities: {}, clientInfo: { name: "tsukkomi-omp-extension", version: "1" } });
      if (this.child !== child) throw new Error("bridge unavailable");
      child.stdin.write(JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" }) + "\n");
    })();
    try { await this.starting; } finally { this.starting = undefined; }
  }

  async tool(name: string, args: object): Promise<Audit> {
    await this.start();
    const response = await this.send("tools/call", { name, arguments: args });
    if (!response || typeof response !== "object" || !("content" in response) || !Array.isArray(response.content) || ("isError" in response && response.isError)) {
      throw new Error("MCP tool failed");
    }
    const part = response.content.find((entry: unknown) => entry && typeof entry === "object" && "type" in entry && entry.type === "text");
    if (!part || typeof part !== "object" || !("text" in part) || typeof part.text !== "string") throw new Error("MCP result missing");
    const parsed: unknown = JSON.parse(part.text);
    if (!parsed || typeof parsed !== "object" || !("ok" in parsed) || parsed.ok !== true ||
      !("verdict" in parsed) || typeof parsed.verdict !== "string" ||
      !["violation", "suspicious", "clean", "unknown"].includes(parsed.verdict)) {
      throw new Error("MCP audit unavailable");
    }
    return parsed as Audit;
  }
}

function session(ctx: Context): Session | undefined {
  try {
    const session_id = ctx.sessionManager.getSessionId();
    const path = ctx.sessionManager.getSessionFile();
    // OMP can publish the path before its first JSONL flush. A prohibited
    // proposed action is still checkable; missing history remains unknown.
    if (!session_id || !path) return undefined;
    return { session_id, provider: "omp", path };
  } catch { return undefined; }
}

function announce(ctx: Context, message: string, level: "info" | "warning" | "error" = "warning"): void {
  try { ctx.ui?.notify(message, level); } catch { /* notifications must not break OMP hooks */ }
}

function description(audit: Audit): string {
  const findings = audit.findings?.filter((finding) => actionable[finding.verdict || ""]) || [];
  const first = findings[0];
  return first ? `Tsukkomi ${audit.verdict}: ${first.contract_id || "contract"}: ${(first.reason || "confirmed contract finding").slice(0, 300)}`
    : `Tsukkomi ${audit.verdict}: confirmed contract finding`;
}

export default function tsukkomiOmp(pi: HookAPI): void {
  if (!enabled) return;
  const bridge = new LocalMcp();
  let lastNotice = "";
  const notifyOnce = (ctx: Context, message: string) => {
    if (message !== lastNotice) { lastNotice = message; announce(ctx, message); }
  };
  const audit = async (ctx: Context) => {
    const current = session(ctx);
    if (!current) { notifyOnce(ctx, "Tsukkomi monitoring unavailable: no OMP session path."); return; }
    try {
      const result = await bridge.tool("audit_session", current);
      if (actionable[result.verdict]) notifyOnce(ctx, description(result));
      else if (result.applicable_contracts === 0) notifyOnce(ctx, "Tsukkomi not applicable: no enabled contract for this OMP session.");
      else if (result.verdict === "unknown") notifyOnce(ctx, "Tsukkomi audit unknown: evidence or verification incomplete.");
      else lastNotice = "";
    } catch { notifyOnce(ctx, "Tsukkomi monitoring unavailable: local MCP bridge/audit failed."); }
  };
  pi.on("session_start", async (_event, ctx) => { lastNotice = ""; await audit(ctx); });
  pi.on("session_switch", async (_event, ctx) => { lastNotice = ""; await audit(ctx); });
  pi.on("tool_call", async (event, ctx) => {
    try {
      const name = event?.toolName;
      // Our stdio bridge never enters OMP's tool pipeline. Skip only manually
      // invoked Tsukkomi MCP tools, not unrelated tools containing that word.
      if (typeof name !== "string" || !name ||
        /^(?:mcp[_:]|mcp__)(?:tsukkomi(?:-mcp)?|self[-_]directing[-_]mcp)(?:__|[_:])/i.test(name)) return;
      const current = session(ctx);
      if (!current) { notifyOnce(ctx, "Tsukkomi monitoring unavailable: no OMP session path."); return; }
      const result = await bridge.tool("check_action", { ...current, action: { tool_name: name, arguments: event.input ?? {} } });
      if (actionable[result.verdict]) return { block: true, reason: description(result) };
      if (result.applicable_contracts === 0) notifyOnce(ctx, "Tsukkomi not applicable: no enabled contract for this OMP session.");
      else if (result.verdict === "unknown") notifyOnce(ctx, "Tsukkomi check unknown: evidence or verification incomplete; action not blocked.");
      else lastNotice = "";
    } catch { notifyOnce(ctx, "Tsukkomi monitoring unavailable: local MCP bridge/check failed; action not blocked."); }
  });
  pi.on("tool_result", async (_event, ctx) => { await audit(ctx); });
  pi.on("turn_end", async (_event, ctx) => { await audit(ctx); });
  pi.on("session_shutdown", () => { bridge.close(); });
}

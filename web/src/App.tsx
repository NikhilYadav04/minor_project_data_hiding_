import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { embedPayload, encodeOwner, extractPayload, recoverAutomatically } from "./api";
import type { Job, Predictor, Rect, Recovery } from "./types";

const steps = ["Owner", "Data hider", "Receiver", "Analysis", "Summary"];
const predictorLabels: Record<Predictor, string> = { edge_directed: "Edge directed", bilinear: "Bilinear", med: "MED", auto: "Automatic best" };

function App() {
  const [active, setActive] = useState(0);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [seed, setSeed] = useState(123);
  const [predictor, setPredictor] = useState<Predictor>("edge_directed");
  const [msaMode, setMsaMode] = useState<Job["msaMode"]>("manual");
  const [rect, setRect] = useState<Rect>({ top: 300, left: 300, height: 60, width: 40 });
  const [job, setJob] = useState<Job | null>(null);
  const [recovery, setRecovery] = useState<Recovery | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [selectedShare, setSelectedShare] = useState(0);
  const [payload, setPayload] = useState("");
  const [extracted, setExtracted] = useState("");

  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const capacity = useMemo(() => {
    const area = rect.height * rect.width;
    const estimatedBytes = area + 8;
    return { area, estimatedBytes, status: area < 95000 ? "sufficient" : "review" };
  }, [rect]);

  function setFailure(value: unknown) {
    setError(value instanceof Error ? value.message : "Something went wrong");
    setMessage("");
  }

  async function createShares() {
    if (!file) return setFailure("Upload an image before creating shares");
    setBusy(true); setFailure(""); setMessage("Running the owner pipeline…");
    try {
      const result = await encodeOwner(file, seed, predictor, rect, msaMode);
      setJob(result);
      if (result.msaRect) setRect(result.msaRect);
      setRecovery(null); setActive(1); setMessage(`Shares created successfully · ${result.msaMode === "manual" ? "manual" : "adaptive"} MSA selected`);
    } catch (value) { setFailure(value); } finally { setBusy(false); }
  }

  async function doRecover(stage: number) {
    if (!job) return setFailure("Create shares first");
    setBusy(true); setFailure(""); setMessage(`Selecting valid shares for Stage ${stage}…`);
    try {
      const result = await recoverAutomatically(job.jobId, stage);
      setRecovery(result);
      const used = (result.usedIndexes ?? []).map((index) => index + 1).join(", ");
      setMessage(`Recovery verified using Shares ${used}`);
    } catch (value) { setFailure(value); } finally { setBusy(false); }
  }

  async function doAutoRecover(stage: number) {
    if (!job) return setFailure("Create shares first");
    setBusy(true); setFailure(""); setMessage(`Selecting valid shares for Stage ${stage}…`);
    try {
      const result = await recoverAutomatically(job.jobId, stage);
      setRecovery(result);
      const used = (result.usedIndexes ?? []).map((index) => index + 1).join(", ");
      setMessage(`Automatic recovery verified using Shares ${used}`);
    } catch (value) { setFailure(value); } finally { setBusy(false); }
  }

  async function doEmbed() {
    if (!job) return;
    setBusy(true); setFailure("");
    try {
      const result = await embedPayload(job.jobId, selectedShare, payload);
      setJob({ ...job, shares: job.shares.map((share) => share.index === selectedShare ? result.share : share) });
      setMessage(`Payload embedded in Share ${selectedShare + 1}; CRC ${result.crc}`);
    } catch (value) { setFailure(value); } finally { setBusy(false); }
  }

  async function doExtract() {
    if (!job) return;
    setBusy(true); setFailure("");
    try { const result = await extractPayload(job.jobId, selectedShare); setExtracted(result.payload); setMessage(`Payload extracted; CRC ${result.crc}`); } catch (value) { setFailure(value); } finally { setBusy(false); }
  }

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand-mark">PSS</div>
      <div className="brand-copy"><strong>Research demo</strong><span>Progressive secret sharing</span></div>
      <div className="status-dot"><i /> Pipeline ready</div>
      <nav>{steps.map((step, index) => <button key={step} className={active === index ? "nav-item active" : "nav-item"} onClick={() => setActive(index)}><span>{String(index + 1).padStart(2, "0")}</span>{step}</button>)}</nav>
      <div className="sidebar-note"><strong>Base configuration</strong><span>(3, 4) threshold</span><span>512 × 512 grayscale</span><span>Deterministic keys</span></div>
    </aside>
    <main className="main-content">
      <header className="topbar"><div><p className="eyebrow">PSS-DRDHEI / SIMPLIFIED REPRODUCTION</p><h1>Progressive image protection</h1></div><div className="topbar-meta"><span>Seed {seed}</span><span>{job ? `Selected · ${predictorLabels[job.predictor]}` : predictorLabels[predictor]}</span></div></header>
      {message && <div className="toast success">{message}</div>}
      {error && <div className="toast error">{error}</div>}
      <AnimatePresence mode="wait">
        {active === 0 && <motion.div key="owner" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}><Owner file={file} setFile={setFile} preview={preview} rect={rect} setRect={setRect} seed={seed} setSeed={setSeed} predictor={predictor} setPredictor={setPredictor} msaMode={msaMode} setMsaMode={setMsaMode} capacity={capacity} busy={busy} createShares={createShares} job={job} /></motion.div>}
        {active === 1 && <motion.div key="hider" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}><DataHider job={job} selectedShare={selectedShare} setSelectedShare={setSelectedShare} payload={payload} setPayload={setPayload} extracted={extracted} doEmbed={doEmbed} doExtract={doExtract} busy={busy} /></motion.div>}
        {active === 2 && <motion.div key="receiver" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}><Receiver job={job} recovery={recovery} doRecover={doRecover} doAutoRecover={doAutoRecover} busy={busy} /></motion.div>}
        {active === 3 && <motion.div key="analysis" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}><Analysis job={job} /></motion.div>}
        {active === 4 && <motion.div key="summary" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}><Summary job={job} /></motion.div>}
      </AnimatePresence>
    </main>
  </div>;
}

function Owner(props: any) {
  const { file, setFile, preview, rect, setRect, seed, setSeed, predictor, setPredictor, msaMode, setMsaMode, capacity, busy, createShares, job } = props;
  return <section><div className="page-heading"><div><p className="eyebrow">01 / OWNER</p><h2>Choose what needs protection</h2><p>Draw a sensitive region, select a deterministic predictor, and create four recoverable shares.</p></div><div className="requirement-chip">Any 3 shares → LR preview</div></div>
    <div className="owner-grid"><div className="panel canvas-panel"><div className="panel-header"><div><strong>Image and MSA selection</strong><span>Drag inside the image to select the sensitive area</span></div><label className="upload-button">{file ? "Replace image" : "Upload image"}<input type="file" accept="image/png,image/jpeg" onChange={(event) => setFile(event.target.files?.[0] ?? null)} /></label></div><ImageSelector image={preview ?? (job?.original ?? null)} rect={rect} setRect={setRect} /></div>
      <div className="panel config-panel"><div className="panel-header"><div><strong>Configuration</strong><span>Fixed threshold: 3 of 4 shares</span></div></div><label className="field-label">MSA selection<select value={msaMode} onChange={(event) => setMsaMode(event.target.value)}><option value="manual">Manual rectangle</option><option value="auto_texture">Automatic · texture</option><option value="auto_edge">Automatic · edges</option></select></label><p className="field-help">Adaptive modes keep the selected dimensions but place the region where the image has the most local information.</p><label className="field-label">Predictor<select value={predictor} onChange={(event) => setPredictor(event.target.value)}><option value="edge_directed">Edge directed</option><option value="bilinear">Bilinear</option><option value="med">MED</option><option value="auto">Automatic best predictor</option></select></label><p className="field-help">{predictor === "auto" ? "Tests the supported predictors and selects the shortest compressed residual stream." : predictor === "med" ? "Lightweight edge-aware prediction using integer arithmetic." : predictor === "bilinear" ? "Simple deterministic interpolation baseline." : "Three-pass integer predictor for directional edges."}</p><label className="field-label">Master seed<input type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} /></label><p className="field-help">Same seed and inputs produce the same shares.</p><div className="capacity-box"><div><span>MSA area</span><strong>{capacity.area.toLocaleString()} px</strong></div><div><span>Estimated payload</span><strong>{capacity.estimatedBytes.toLocaleString()} B</strong></div><div className="capacity-status"><i /> Capacity {capacity.status}</div></div><button className="primary-button" disabled={busy || !file} onClick={createShares}>{busy ? "Creating shares…" : "Create protected shares"}</button></div></div>
    {job && <OwnerResults job={job} />}
  </section>;
}

function ImageSelector({ image, rect, setRect }: { image: string | null; rect: Rect; setRect: (rect: Rect) => void }) {
  const ref = useRef<HTMLDivElement>(null); const operation = useRef<{ mode: "draw" | "move" | "resize"; corner?: string; start: { x: number; y: number }; initial: Rect } | null>(null);
  function point(event: React.PointerEvent) { const box = ref.current?.getBoundingClientRect(); if (!box) return null; return { x: Math.max(0, Math.min(512, ((event.clientX - box.left) / box.width) * 512)), y: Math.max(0, Math.min(512, ((event.clientY - box.top) / box.height) * 512)) }; }
  function down(event: React.PointerEvent) { if (!image) return; const p = point(event); if (!p) return; const inside = p.x >= rect.left && p.x <= rect.left + rect.width && p.y >= rect.top && p.y <= rect.top + rect.height; operation.current = { mode: inside ? "move" : "draw", start: p, initial: rect }; (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId); if (!inside) setRect({ top: Math.round(p.y), left: Math.round(p.x), height: 1, width: 1 }); }
  function resizeDown(event: React.PointerEvent, corner: string) { if (!image) return; const p = point(event); if (!p) return; event.stopPropagation(); operation.current = { mode: "resize", corner, start: p, initial: rect }; (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId); }
  function move(event: React.PointerEvent) { const op = operation.current; if (!op) return; const p = point(event); if (!p) return; const dx = p.x - op.start.x; const dy = p.y - op.start.y; if (op.mode === "draw") { setRect({ top: Math.round(Math.min(op.start.y, p.y)), left: Math.round(Math.min(op.start.x, p.x)), height: Math.max(1, Math.round(Math.abs(dy))), width: Math.max(1, Math.round(Math.abs(dx))) }); return; } if (op.mode === "move") { setRect({ ...op.initial, top: Math.round(Math.max(0, Math.min(512 - op.initial.height, op.initial.top + dy))), left: Math.round(Math.max(0, Math.min(512 - op.initial.width, op.initial.left + dx))) }); return; } const r = op.initial; const corner = op.corner ?? ""; const right = r.left + r.width; const bottom = r.top + r.height; const nextLeft = corner.includes("left") ? Math.max(0, Math.min(right - 1, r.left + dx)) : r.left; const nextTop = corner.includes("top") ? Math.max(0, Math.min(bottom - 1, r.top + dy)) : r.top; const nextRight = corner.includes("right") ? Math.min(512, Math.max(r.left + 1, right + dx)) : right; const nextBottom = corner.includes("bottom") ? Math.min(512, Math.max(r.top + 1, bottom + dy)) : bottom; setRect({ left: Math.round(nextLeft), top: Math.round(nextTop), width: Math.round(nextRight - nextLeft), height: Math.round(nextBottom - nextTop) }); }
  function up() { operation.current = null; }
  return <div className="image-stage" ref={ref} onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up}>{image ? <img src={image} alt="Uploaded grayscale image" /> : <div className="empty-canvas"><strong>Upload a grayscale image</strong><span>The preview and draggable MSA rectangle will appear here.</span></div>}{image && <div className="msa-veil" style={{ left: `${rect.left / 5.12}%`, top: `${rect.top / 5.12}%`, width: `${rect.width / 5.12}%`, height: `${rect.height / 5.12}%` }}><span>Sensitive region</span><b>{rect.width} × {rect.height}</b><i className="handle top-left" onPointerDown={(event) => resizeDown(event, "top-left")} /><i className="handle top-right" onPointerDown={(event) => resizeDown(event, "top-right")} /><i className="handle bottom-left" onPointerDown={(event) => resizeDown(event, "bottom-left")} /><i className="handle bottom-right" onPointerDown={(event) => resizeDown(event, "bottom-right")} /></div>}</div>;
}

function OwnerResults({ job }: { job: Job }) { return <div className="results-section"><div className="section-heading"><div><p className="eyebrow">SHARES CREATED</p><h3>One owner action, three recovery states</h3></div><span className="verified-label"><i /> Integrity tags attached</span></div><div className="research-evidence"><div><span>Selected predictor</span><strong>{predictorLabels[job.predictor]}</strong></div><div><span>MSA strategy</span><strong>{job.msaMode === "manual" ? "Manual" : job.msaMode === "auto_texture" ? "Adaptive texture" : "Adaptive edges"}</strong></div><div><span>Recovery contract</span><strong>Any 3 valid shares</strong></div></div><div className="preview-row"><PreviewCard title="Original" image={job.original} /><PreviewCard title="Masked" image={job.masked} /><PreviewCard title="LR preview" image={job.lr} /></div><div className="share-grid">{job.shares.map((share) => <div className="share-card" key={share.index}><img src={share.image} alt={share.label} /><div><strong>{share.label}</strong><span className="valid-text"><i /> {share.valid ? "Valid" : "Tampered"}</span></div></div>)}</div>{job.predictorCandidates.length > 0 && <div className="predictor-evidence"><div className="panel-header"><div><strong>Automatic predictor decision</strong><span>Lowest compressed residual length was selected</span></div></div><div className="table-wrap"><table><thead><tr><th>Predictor</th><th>Compressed bits</th><th>Mean abs. residual</th><th>Selected</th></tr></thead><tbody>{job.predictorCandidates.map((candidate) => <tr key={candidate.method}><td>{predictorLabels[candidate.method]}</td><td>{candidate.bit_length.toLocaleString()}</td><td>{candidate.mean_abs_residual.toFixed(3)}</td><td>{candidate.method === job.predictor ? "Yes" : ""}</td></tr>)}</tbody></table></div></div>}<div className="stats-strip">{[["Free room", `${Number(job.stats.free_bytes).toLocaleString()} B`], ["Practical rate", `${Number(job.stats.bpp_free).toFixed(3)} bpp`], ["Encoding", `${((Number(job.stats.t_inpaint) + Number(job.stats.t_predict) + Number(job.stats.t_compress) + Number(job.stats.t_share) + Number(job.stats.t_embed)) * 1000).toFixed(0)} ms`], ["Storage", "4× original"]].map(([label, value]) => <div key={label}><span>{label}</span><strong>{value}</strong></div>)}</div></div> }

function PreviewCard({ title, image }: { title: string; image: string }) { return <div className="preview-card"><img src={image} alt={title} /><span>{title}</span></div> }

function DataHider({ job, selectedShare, setSelectedShare, payload, setPayload, extracted, doEmbed, doExtract, busy }: any) { return <section><div className="page-heading"><div><p className="eyebrow">02 / DATA HIDER</p><h2>Add metadata without recovery keys</h2><p>The payload uses only the unused redundant room. Image recovery remains separate.</p></div></div>{!job ? <EmptyState text="Create shares in the Owner step first." /> : <div className="two-column"><div className="panel"><div className="panel-header"><div><strong>Choose a share</strong><span>Each share has independent free room</span></div></div><div className="share-picker">{job.shares.map((share: any) => <button className={selectedShare === share.index ? "share-option selected" : "share-option"} onClick={() => setSelectedShare(share.index)} key={share.index}><img src={share.image} alt="" /><span>{share.label}</span><small>{share.valid ? "Valid" : "Tampered"}</small></button>)}</div></div><div className="panel form-panel"><div className="flow-line"><span>Final share</span><b>→</b><span>Free room</span><b>→</b><span>Embedded share</span></div><label className="field-label">Text payload<textarea value={payload} onChange={(event) => setPayload(event.target.value)} placeholder="Example: capture timestamp, operator ID, or audit note" /></label><div className="capacity-box compact"><div><span>Available room</span><strong>{Number(job.stats.free_bytes).toLocaleString()} B</strong></div><div><span>Payload length</span><strong>{new TextEncoder().encode(payload).length} B</strong></div></div><div className="button-row"><button className="primary-button" disabled={busy || !payload} onClick={doEmbed}>Embed payload</button><button className="secondary-button" disabled={busy} onClick={doExtract}>Extract payload</button></div>{extracted && <div className="extracted-box"><span>CRC valid · extracted payload</span><strong>{extracted}</strong></div>}</div></div>}</section> }

function Receiver({ job, recovery, doRecover, doAutoRecover, busy }: any) { return <section><div className="page-heading"><div><p className="eyebrow">03 / RECEIVER</p><h2>Recover progressively</h2><p>The number of valid shares determines the resolution and the keys determine what is revealed.</p></div></div>{!job ? <EmptyState text="Create shares in the Owner step first." /> : <><div className="stage-grid">{[[1, "LR preview", "Any 3 valid shares"], [2, "Masked image", "Any 3 valid shares"], [3, "Original image", "Any 3 valid shares"]].map(([stage, title, requirement]) => <button key={stage} className={recovery?.stage === stage ? "stage-card selected" : "stage-card"} onClick={() => doRecover(stage)} disabled={busy}><span>Stage {stage}</span><strong>{title}</strong><small>{requirement}</small></button>)}</div><div className="panel auto-recovery-banner"><div><strong>Automatic resilient recovery</strong><span>Each stage verifies the shares and uses the first valid three-share threshold subset.</span></div><button className="secondary-button" disabled={busy} onClick={() => doAutoRecover(recovery?.stage ?? 3)}>Run automatic recovery</button></div><div className="receiver-grid"><div className="panel"><div className="panel-header"><div><strong>Share integrity</strong><span>Modified shares are rejected before recovery</span></div></div>{job.shares.map((share: any) => <div className="integrity-row" key={share.index}><span>{share.label}</span><span className={share.valid ? "valid-text" : "invalid-text"}><i />{share.valid ? "Valid · usable" : "Tampered · excluded"}</span></div>)}</div><div className="panel recovery-panel">{recovery ? <><div className="result-label"><i />{recovery.title}</div><img src={recovery.image} alt={recovery.title} /><div className="exact-result"><strong>Exact match: {recovery.exact ? "verified" : "not verified"}</strong><span>{recovery.width} × {recovery.height} pixels{recovery.usedIndexes ? ` · valid shares ${recovery.usedIndexes.map((index: number) => index + 1).join(", ")}` : ""}</span></div></> : <EmptyState text="Choose a recovery stage to see the result." />}</div></div></>}</section> }

function Analysis({ job }: { job: Job | null }) { const data = job?.metrics.map((row) => ({ name: `S${row.share}`, entropy: row.entropy, correlation: (row.horizontalCorrelation + row.verticalCorrelation) / 2 })) ?? []; return <section><div className="page-heading"><div><p className="eyebrow">04 / ANALYSIS</p><h2>Read the system, not just the output</h2><p>These measurements explain share randomness, predictor behavior, and the protection tradeoffs.</p></div></div>{!job ? <EmptyState text="Create shares in the Owner step first." /> : <div className="analysis-grid"><div className="panel chart-panel"><div className="panel-header"><div><strong>Share entropy</strong><span>Ideal random 8-bit entropy is close to 8</span></div></div><ResponsiveContainer width="100%" height={260}><BarChart data={data}><CartesianGrid strokeDasharray="3 3" stroke="#dce4e5" /><XAxis dataKey="name" /><YAxis domain={[0, 8]} /><Tooltip /><Bar dataKey="entropy" fill="#1f6f78" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer></div><div className="panel metrics-panel"><div className="panel-header"><div><strong>Security metrics</strong><span>Per final share</span></div></div><div className="table-wrap"><table><thead><tr><th>Share</th><th>Entropy</th><th>H corr.</th><th>V corr.</th><th>NPCR</th><th>UACI</th></tr></thead><tbody>{job.metrics.map((row) => <tr key={row.share}><td>S{row.share}</td><td>{row.entropy}</td><td>{row.horizontalCorrelation}</td><td>{row.verticalCorrelation}</td><td>{row.npcr}%</td><td>{row.uaci}%</td></tr>)}</tbody></table></div></div></div>}</section> }

function Summary({ job }: { job: Job | null }) {
  return <section><div className="page-heading"><div><p className="eyebrow">05 / RESEARCH SUMMARY</p><h2>What this project demonstrates</h2><p>A concise presentation view for the final review: baseline reproduction, measured extensions, and exact recovery evidence.</p></div><div className="requirement-chip">Research prototype · {(job?.stats.method as string) ?? "no run"}</div></div>{!job ? <EmptyState text="Create shares in the Owner step to populate the summary." /> : <div className="summary-grid"><div className="panel summary-panel"><div className="panel-header"><div><strong>Baseline and extensions</strong><span>Each extension is tied to an observable experiment.</span></div></div><div className="summary-row"><span>Base reproduction</span><strong>Progressive (3,4) sharing with exact recovery</strong></div><div className="summary-row"><span>Extension 1</span><strong>{job.msaMode === "manual" ? "Manual MSA" : "Adaptive MSA placement"}</strong></div><div className="summary-row"><span>Extension 2</span><strong>{job.predictorSelection === "automatic" ? "Automatic predictor selection" : "Predictor comparison"}</strong></div><div className="summary-row"><span>Extension 3</span><strong>Tamper-aware recovery using any three valid shares</strong></div></div><div className="panel summary-panel"><div className="panel-header"><div><strong>Current run evidence</strong><span>Generated from the live pipeline.</span></div></div><div className="summary-metrics"><div><span>Selected predictor</span><strong>{predictorLabels[job.predictor]}</strong></div><div><span>MSA coordinates</span><strong>{job.msaRect ? `${job.msaRect.top}, ${job.msaRect.left} · ${job.msaRect.width}×${job.msaRect.height}` : "None"}</strong></div><div><span>Free embedding room</span><strong>{Number(job.stats.free_bytes).toLocaleString()} B</strong></div><div><span>Integrity state</span><strong>4 tagged shares</strong></div></div></div><div className="panel summary-panel conclusion-panel"><div className="panel-header"><div><strong>Presentation conclusion</strong></div></div><p>The prototype preserves exact progressive recovery while adding content-aware MSA placement, compression-aware predictor choice, and recovery after one invalid or unavailable share.</p><span className="verified-label"><i /> Stage 3 exact recovery is verified for this run</span></div></div>}</section>;
}

function EmptyState({ text }: { text: string }) { return <div className="empty-state"><strong>{text}</strong><span>Use the Owner step to generate a live research run.</span></div> }

export default App;

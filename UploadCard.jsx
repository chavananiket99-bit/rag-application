import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { CheckCircle2, FileText, Loader2, UploadCloud, X } from "lucide-react";
import { uploadPdf } from "../services/api";
import { useUpload } from "../context/UploadContext";

// Share of the bar covered by the real network transfer; the rest tracks server processing.
const TRANSFER_SHARE = 25;
const PROCESSING_CEILING = 96;
// Server side reports no progress, so the bar eases toward the ceiling on this time constant.
const PROCESSING_TIME_CONSTANT_SECONDS = 180;

function formatDuration(totalSeconds) {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
}

export default function UploadCard() {
  const { refreshDocuments } = useUpload();
  const navigate = useNavigate();
  const fileInputRef = useRef(null);
  const processingTimerRef = useRef(null);
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("");
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  const stopProcessingTimer = () => {
    if (processingTimerRef.current) {
      clearInterval(processingTimerRef.current);
      processingTimerRef.current = null;
    }
  };

  useEffect(() => stopProcessingTimer, []);

  const startProcessingTimer = () => {
    if (processingTimerRef.current) {
      return;
    }
    const startedAt = Date.now();
    setStatus("Extracting text and building AI knowledge...");
    processingTimerRef.current = setInterval(() => {
      const seconds = (Date.now() - startedAt) / 1000;
      setElapsedSeconds(Math.floor(seconds));
      const completed =
        1 - Math.exp(-seconds / PROCESSING_TIME_CONSTANT_SECONDS);
      setProgress(
        Math.round(
          TRANSFER_SHARE + (PROCESSING_CEILING - TRANSFER_SHARE) * completed,
        ),
      );
    }, 500);
  };

  const onFileChange = (event) => {
    const selectedFiles = Array.from(event.target.files || []);
    const pdfFiles = selectedFiles.filter(
      (file) => file.type === "application/pdf",
    );
    setFiles(pdfFiles);
    setResult(null);
    setError("");
    setProgress(0);
    setStatus("");
    setElapsedSeconds(0);
  };
  const removeFile = (index) => {
    setFiles((previous) => previous.filter((_, i) => i !== index));
    setResult(null);
    setError("");
  };
  const onUpload = async () => {
    if (!files || files.length === 0) {
      setError("Please select at least one PDF document.");
      return;
    }
    try {
      setLoading(true);
      setError("");
      setResult(null);
      setProgress(0);
      setElapsedSeconds(0);
      setStatus("Uploading document...");
      // Expiry and agreement type are detected from the document text by the backend.
      const data = await uploadPdf(files, [], (event) => {
        if (!event.total) {
          startProcessingTimer();
          return;
        }
        if (event.loaded >= event.total) {
          setProgress(TRANSFER_SHARE);
          startProcessingTimer();
          return;
        }
        setProgress(Math.round((event.loaded / event.total) * TRANSFER_SHARE));
      });
      stopProcessingTimer();
      setResult(data);
      setProgress(PROCESSING_CEILING);
      setStatus("Refreshing knowledge base...");
      /*
       * Refresh normalized documents from backend.
       */
      await refreshDocuments();
      setProgress(100);
      setStatus("Document uploaded successfully.");
      /*
       * Move to Chat after successful upload.
       */
      setTimeout(() => {
        navigate("/chat");
      }, 900);
    } catch (err) {
      const message =
        err?.response?.data?.detail ||
        err?.message ||
        "Upload failed. Please check the backend server.";
      setError(String(message));
      setProgress(0);
      setStatus("");
    } finally {
      stopProcessingTimer();
      setLoading(false);
    }
  };
  return (
    <div className="w-full max-w-3xl">
      {/* Upload area */}
      <div
        className={`
         relative overflow-hidden rounded-2xl
         border-2 border-dashed
         bg-slate-50
         p-6 sm:p-8
         transition-all duration-200
         ${
           loading
             ? "border-red-300 bg-red-50/30"
             : "border-slate-300 hover:border-red-400 hover:bg-red-50/20"
         }
       `}
      >
        {/* Decorative background */}
        <div className="pointer-events-none absolute -right-16 -top-16 h-40 w-40 rounded-full bg-red-100/50 blur-3xl" />
        <div className="relative">
          {/* Icon */}
          <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-red-50">
            {loading ? (
              <Loader2 size={30} className="animate-spin text-red-700" />
            ) : (
              <UploadCloud size={30} className="text-red-700" />
            )}
          </div>
          {/* Heading */}
          <div className="mt-5 text-center">
            <h3 className="text-lg font-semibold text-slate-800">
              Upload your PDF documents
            </h3>
            <p className="mx-auto mt-2 max-w-lg text-sm leading-6 text-slate-500">
              Add enterprise documents to your knowledge base and make them
              available to the AI assistant.
            </p>
          </div>
          {/* Select button */}
          <div className="mt-6 flex justify-center">
            <button
              type="button"
              disabled={loading}
              onClick={() => fileInputRef.current?.click()}
              className="
               inline-flex items-center gap-2
               rounded-xl
               bg-red-700
               px-5 py-3
               text-sm font-semibold text-white
               shadow-sm
               transition
               hover:bg-red-800
               disabled:cursor-not-allowed
               disabled:opacity-50
             "
            >
              <UploadCloud size={18} />
              Select PDF
            </button>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept="application/pdf"
              onChange={onFileChange}
              className="hidden"
            />
          </div>
          <p className="mt-3 text-center text-xs text-slate-400">
            PDF files only
          </p>
          {/* Selected files */}
          {files.length > 0 && (
            <div className="mt-7">
              <div className="mb-3 flex items-center justify-between">
                <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Selected document
                  {files.length > 1 ? "s" : ""}
                </p>
                <span className="rounded-full bg-slate-200 px-2.5 py-1 text-[11px] font-medium text-slate-600">
                  {files.length}
                </span>
              </div>
              <div className="space-y-2">
                {files.map((file, index) => (
                  <div
                    key={`${file.name}-${index}`}
                    className="flex items-center gap-3 rounded-xl border border-slate-200 bg-white px-4 py-3 shadow-sm"
                  >
                    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-red-50">
                      <FileText size={19} className="text-red-700" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <p
                        className="truncate text-sm font-medium text-slate-700"
                        title={file.name}
                      >
                        {file.name}
                      </p>
                      <p className="mt-0.5 text-xs text-slate-400">
                        {(file.size / 1024 / 1024).toFixed(2)} MB · PDF
                      </p>
                    </div>
                    {!loading && (
                      <button
                        type="button"
                        onClick={() => removeFile(index)}
                        className="rounded-lg p-2 text-slate-400 transition hover:bg-red-50 hover:text-red-600"
                        title="Remove document"
                      >
                        <X size={17} />
                      </button>
                    )}
                  </div>
                ))}
              </div>
              {/* Upload button */}
              <button
                type="button"
                onClick={onUpload}
                disabled={loading}
                className="
                 mt-4
                 flex w-full
                 items-center justify-center gap-2
                 rounded-xl
                 bg-slate-900
                 px-5 py-3
                 text-sm font-semibold text-white
                 shadow-sm
                 transition
                 hover:bg-slate-800
                 disabled:cursor-not-allowed
                 disabled:opacity-50
               "
              >
                {loading ? (
                  <>
                    <Loader2 size={18} className="animate-spin" />
                    Processing document...
                  </>
                ) : (
                  <>
                    <UploadCloud size={18} />
                    Upload to Knowledge Base
                  </>
                )}
              </button>
            </div>
          )}
          {/* Progress */}
          {loading && (
            <div className="mt-6 rounded-xl border border-slate-200 bg-white p-4">
              <div className="mb-2 flex items-center justify-between">
                <span className="text-xs font-medium text-slate-600">
                  {status}
                </span>
                <span className="text-xs font-semibold text-red-700">
                  {progress}%
                </span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-slate-100">
                <div
                  className="h-full rounded-full bg-red-700 transition-all duration-500"
                  style={{
                    width: `${progress}%`,
                  }}
                />
              </div>
              {elapsedSeconds > 0 && (
                <p className="mt-2 text-[11px] text-slate-400">
                  {formatDuration(elapsedSeconds)} elapsed · scanned PDFs are
                  processed page by page and can take several minutes
                </p>
              )}
            </div>
          )}
          {/* Success */}
          {result && !loading && (
            <div className="mt-6 rounded-xl border border-emerald-200 bg-emerald-50 p-4">
              <div className="flex items-start gap-3">
                <CheckCircle2
                  size={20}
                  className="mt-0.5 shrink-0 text-emerald-600"
                />
                <div className="min-w-0">
                  <p className="text-sm font-semibold text-emerald-800">
                    Document uploaded successfully
                  </p>
                  <p className="mt-1 text-xs leading-5 text-emerald-700">
                    {result.successful_files} document
                    {result.successful_files === 1 ? "" : "s"} successfully
                    indexed.
                  </p>
                  {result.failed_files > 0 && (
                    <p className="mt-2 text-[11px] text-emerald-600">
                      {result.failed_files} document
                      {result.failed_files === 1 ? "" : "s"} failed to upload.
                    </p>
                  )}
                </div>
              </div>
            </div>
          )}
          {/* Error */}
          {error && (
            <div className="mt-5 rounded-xl border border-red-200 bg-red-50 px-4 py-3">
              <p className="text-sm text-red-700">{error}</p>
            </div>
          )}
        </div>
      </div>
      {/* Information below upload box */}
      <div className="mt-4 flex flex-col gap-2 text-center text-[11px] text-slate-400 sm:flex-row sm:items-center sm:justify-center sm:gap-5">
        <span>✓ Secure document processing</span>
        <span>✓ AI-powered document search</span>
        <span>✓ Enterprise knowledge base</span>
      </div>
    </div>
  );
}

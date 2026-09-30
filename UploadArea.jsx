import { Upload, FileText } from "lucide-react";

export default function UploadArea({ file, onChange, onUpload, loading }) {
  return (
    <div className="mx-auto max-w-3xl rounded-2xl border-2 border-dashed border-red-300 bg-white p-10 shadow">
      <div className="flex flex-col items-center">
        <Upload className="text-red-700" size={60} />
        <h2 className="mt-5 text-2xl font-bold text-gray-800">
          Upload Documents
        </h2>
        <p className="mt-2 text-gray-500 text-center">
          Upload one or more PDF files to chat with your documents.
        </p>

        <input
          type="file"
          multiple
          accept=".pdf"
          onChange={onChange}
          className="mt-8"
        />

        {file?.length > 0 && (
          <div className="mt-6 w-full">
            {Array.from(file).map((f, index) => (
              <div
                key={index}
                className="flex items-center gap-3 rounded-lg border p-3 mt-2"
              >
                <FileText className="text-red-700" size={20} />
                <span>{f.name}</span>
              </div>
            ))}
          </div>
        )}

        <button
          onClick={onUpload}
          disabled={loading}
          className="mt-8 rounded-lg bg-red-700 px-8 py-3 text-white hover:bg-red-800"
        >
          {loading ? "Uploading..." : "Upload"}
        </button>
      </div>
    </div>
  );
}

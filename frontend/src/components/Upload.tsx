/**
 * Upload Component
 * 
 * Drag-and-drop CSV file upload with "Use Sample Data" option.
 * Includes session switcher for returning users.
 * Follows design system: warm neutral palette, no visual noise.
 */

import { useState, useCallback, DragEvent, ChangeEvent } from 'react';
import { Upload as UploadIcon, FileText, Loader2 } from 'lucide-react';
import type { Session } from '../types';

interface UploadProps {
  onFileUpload: (file: File) => Promise<void>;
  onUseSampleData: () => Promise<void>;
  isLoading: boolean;
  error: string | null;
  sessions?: Session[];
  onSwitchSession?: (sessionId: string) => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  user?: any;
}

export function Upload({
  onFileUpload,
  onUseSampleData,
  isLoading,
  error,
  sessions = [],
  onSwitchSession,
  user
}: UploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  // Drag handlers
  const handleDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.csv')) {
      setSelectedFile(file);
    }
  }, []);

  // File input handler
  const handleFileChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
    }
  }, []);

  // Submit handler
  const handleSubmit = useCallback(async () => {
    if (selectedFile) {
      await onFileUpload(selectedFile);
    }
  }, [selectedFile, onFileUpload]);

  return (
    <div className="min-h-screen bg-bg flex items-center justify-center px-4 sm:px-6">
      <div className="w-full max-w-md">
        {/* Header with personalized greeting */}
        <div className="text-center mb-8">
          {user?.username ? (
            <>
              <h1 className="text-xl sm:text-2xl font-semibold text-textPrimary mb-2">
                Hey {user.username}! 👋
              </h1>
              <p className="text-sm text-muted">
                Ready to take control of your finances? Let's dive in.
              </p>
            </>
          ) : (
            <>
              <h1 className="text-xl sm:text-2xl font-semibold text-textPrimary mb-2">
                Your Financial Coach
              </h1>
              <p className="text-sm text-muted">
                Upload your bank statement to get personalized insights
              </p>
            </>
          )}
        </div>

        {/* Upload Card */}
        <div className="bg-surface rounded-2xl p-6 sm:p-8 shadow-sm border border-border/40">
          {/* Drag-Drop Zone */}
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`
              relative
              border-2 border-dashed rounded-xl p-6 sm:p-8
              flex flex-col items-center justify-center
              transition-all duration-150
              cursor-pointer
              ${isDragging
                ? 'border-[#A8D8EA] bg-[#E8F4F8]'
                : selectedFile
                  ? 'border-[#B5EAD7] bg-[#F0FAF6]'
                  : 'border-border/60 hover:border-[#A8D8EA] hover:bg-[#F8FBFC]'
              }
            `}
          >
            {selectedFile ? (
              <>
                <div className="w-12 h-12 rounded-full bg-[#B5EAD7] flex items-center justify-center mb-3">
                  <FileText className="w-6 h-6 text-[#155724]" />
                </div>
                <p className="text-sm font-medium text-textPrimary mb-0.5">
                  {selectedFile.name}
                </p>
                <p className="text-xs text-muted">
                  {(selectedFile.size / 1024).toFixed(1)} KB
                </p>
              </>
            ) : (
              <>
                <div className="w-12 h-12 rounded-full bg-[#E8F4F8] flex items-center justify-center mb-3">
                  <UploadIcon className="w-6 h-6 text-[#1A5276]" />
                </div>
                <p className="text-sm text-textSecondary mb-0.5">
                  Drag and drop your CSV file here
                </p>
                <p className="text-xs text-muted">
                  or click to browse
                </p>
              </>
            )}

            {/* Hidden file input - positioned absolutely within relative parent */}
            <input
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
          </div>

          {/* Error Message */}
          {error && (
            <div className="mt-4 p-3 bg-bg rounded-xl">
              <p className="text-sm text-textSecondary">{error}</p>
            </div>
          )}

          {/* Action Buttons */}
          <div className="mt-5 space-y-3">
            {/* Upload Button */}
            <button
              onClick={handleSubmit}
              disabled={!selectedFile || isLoading}
              className={`
                w-full px-4 py-2.5 rounded-xl text-sm font-medium
                transition-all duration-150
                ${selectedFile && !isLoading
                  ? 'bg-[#A8D8EA] text-[#1A5276] hover:bg-[#8BC4D9]'
                  : 'bg-bg text-muted cursor-not-allowed'
                }
              `}
            >
              {isLoading ? (
                <span className="flex items-center justify-center">
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Analyzing...
                </span>
              ) : (
                'Analyze Transactions'
              )}
            </button>

            {/* Divider */}
            <div className="flex items-center">
              <div className="flex-1 border-t border-border/60" />
              <span className="px-4 text-xs text-muted">or</span>
              <div className="flex-1 border-t border-border/60" />
            </div>

            {/* Sample Data Button */}
            <button
              onClick={onUseSampleData}
              disabled={isLoading}
              className={`
                w-full px-4 py-2.5 rounded-xl text-sm font-medium
                bg-[#F5E6F3] text-[#5B2C6F] border border-[#DDA0DD]/30
                transition-all duration-150
                ${isLoading ? 'opacity-50 cursor-not-allowed' : 'hover:bg-[#ECD8EA]'}
              `}
            >
              Use Sample Data
            </button>
          </div>

          {/* Help Text */}
          <p className="mt-5 text-xs text-muted text-center">
            Your data is processed locally and never stored permanently.
          </p>
        </div>

        {/* Format Hint */}
        <div className="mt-5 text-center">
          <p className="text-[11px] text-muted">
            CSV format: date, description, amount
          </p>
        </div>
      </div>
    </div>
  );
}


import React, { useState } from 'react';
import { generateCodeSnippet } from '../services/geminiService';

const CodePlayground: React.FC = () => {
  const [task, setTask] = useState('Create a simple neural network in PyTorch');
  const [result, setResult] = useState<{ code: string, explanation: string } | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleGenerate = async () => {
    setIsLoading(true);
    const data = await generateCodeSnippet(task);
    setResult(data);
    setIsLoading(false);
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
      <div className="bg-slate-800/50 p-6 border-b border-slate-800">
        <h2 className="text-xl font-bold text-white mb-4">Interactive AI Playground</h2>
        <div className="flex flex-col md:flex-row gap-3">
          <input
            type="text"
            value={task}
            onChange={(e) => setTask(e.target.value)}
            className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-4 py-3 text-slate-200 focus:outline-none focus:border-blue-500 text-sm"
            placeholder="What AI function do you want to build?"
          />
          <button
            onClick={handleGenerate}
            disabled={isLoading}
            className="bg-blue-600 hover:bg-blue-500 text-white px-6 py-3 rounded-lg font-bold transition-all flex items-center justify-center gap-2 min-w-[140px]"
          >
            {isLoading ? (
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                Generate
              </>
            )}
          </button>
        </div>
      </div>

      <div className="grid md:grid-cols-2 min-h-[400px]">
        <div className="bg-slate-950 p-6 border-r border-slate-800 overflow-x-auto">
          <div className="flex items-center gap-2 mb-4">
            <div className="w-3 h-3 rounded-full bg-red-500"></div>
            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
            <div className="w-3 h-3 rounded-full bg-green-500"></div>
            <span className="text-xs text-slate-500 ml-2 font-mono">ai_output.py</span>
          </div>
          <pre className="code-font text-sm text-blue-300 whitespace-pre-wrap">
            {result?.code || "# Enter a task above to see the AI generate code...\n\n# Example: 'Scrape news using Gemini API'\n# Example: 'Text summarization with Transformers'"}
          </pre>
        </div>
        <div className="bg-slate-900 p-6 flex flex-col">
          <h4 className="text-xs uppercase tracking-widest text-slate-500 font-bold mb-4">Explanation & Insights</h4>
          {result ? (
            <div className="text-slate-300 text-sm leading-relaxed overflow-y-auto">
              {result.explanation}
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center text-slate-600 italic text-sm">
              Waiting for input...
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CodePlayground;

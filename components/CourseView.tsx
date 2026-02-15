
import React, { useState, useEffect } from 'react';
import { Course, Lesson } from '../types';
import { generateExplanation } from '../services/geminiService';

interface CourseViewProps {
  course: Course;
  onBack: () => void;
  completedLessonIds: string[];
  onToggleCompletion: (lessonId: string) => void;
}

// A lightweight Markdown renderer for the application
const MarkdownRenderer: React.FC<{ content: string }> = ({ content }) => {
  if (!content) return null;

  // Split by code blocks first
  const blocks = content.split(/(```[\s\S]*?```)/g);

  return (
    <div className="space-y-4">
      {blocks.map((block, index) => {
        if (block.startsWith('```')) {
          // Code Block
          const code = block.replace(/```(\w+)?\n?/, '').replace(/```$/, '').trim();
          const language = block.match(/```(\w+)/)?.[1] || 'code';
          return (
            <div key={index} className="my-6 rounded-xl overflow-hidden border border-slate-700 bg-slate-950 shadow-2xl">
              <div className="flex items-center justify-between px-4 py-2 bg-slate-900 border-b border-slate-800">
                <span className="text-[10px] font-bold uppercase tracking-widest text-slate-500">{language}</span>
                <button 
                  onClick={() => navigator.clipboard.writeText(code)}
                  className="text-slate-500 hover:text-blue-400 transition-colors"
                  title="Copy to clipboard"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
                  </svg>
                </button>
              </div>
              <pre className="p-4 overflow-x-auto code-font text-sm leading-relaxed text-blue-300">
                <code>{code}</code>
              </pre>
            </div>
          );
        }

        // Regular Text Parsing (Headers, Bold, Inline Code, Lists)
        const lines = block.split('\n');
        return (
          <div key={index} className="text-slate-300 leading-relaxed font-sans">
            {lines.map((line, lIdx) => {
              // Header 3
              if (line.startsWith('### ')) {
                return <h3 key={lIdx} className="text-xl font-bold text-white mt-8 mb-4 border-l-4 border-blue-500 pl-4">{line.replace('### ', '')}</h3>;
              }
              // Header 2
              if (line.startsWith('## ')) {
                return <h2 key={lIdx} className="text-2xl font-bold text-white mt-10 mb-6">{line.replace('## ', '')}</h2>;
              }
              // Bullet points
              if (line.trim().startsWith('* ') || line.trim().startsWith('- ')) {
                const text = line.trim().substring(2);
                return (
                  <div key={lIdx} className="flex items-start gap-3 my-2 ml-4">
                    <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-blue-500 shrink-0" />
                    <span className="text-slate-300 italic-inline-code">{parseInline(text)}</span>
                  </div>
                );
              }
              // Normal paragraph line
              return <p key={lIdx} className={`${line.trim() === '' ? 'h-4' : 'mb-2'}`}>{parseInline(line)}</p>;
            })}
          </div>
        );
      })}
    </div>
  );
};

// Helper to parse inline styles: bold and inline code
const parseInline = (text: string) => {
  const parts = text.split(/(\*\*.*?\*\*|`.*?`)/g);
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={i} className="text-white font-bold">{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith('`') && part.endsWith('`')) {
      return <code key={i} className="bg-slate-800 px-1.5 py-0.5 rounded text-blue-400 code-font text-xs mx-0.5">{part.slice(1, -1)}</code>;
    }
    return part;
  });
};

const CourseView: React.FC<CourseViewProps> = ({ course, onBack, completedLessonIds, onToggleCompletion }) => {
  const [activeLesson, setActiveLesson] = useState<Lesson | null>(course.lessonsList?.[0] || null);
  const [lessonExplanation, setLessonExplanation] = useState<string>('');
  const [isGenerating, setIsGenerating] = useState(false);

  const currentIndex = course.lessonsList?.findIndex(l => l.id === activeLesson?.id) ?? -1;
  const isLastLesson = currentIndex === (course.lessonsList?.length ?? 0) - 1;
  const isLessonCompleted = activeLesson ? completedLessonIds.includes(activeLesson.id) : false;
  
  const progressPercent = Math.round((completedLessonIds.length / (course.lessonsList?.length || 1)) * 100);

  useEffect(() => {
    if (activeLesson) {
      fetchExplanation(activeLesson.title);
    }
  }, [activeLesson]);

  const fetchExplanation = async (topic: string) => {
    setIsGenerating(true);
    setLessonExplanation('');
    const explanation = await generateExplanation(topic);
    setLessonExplanation(explanation);
    setIsGenerating(false);
  };

  const handleNext = () => {
    if (currentIndex !== -1 && course.lessonsList && !isLastLesson) {
      setActiveLesson(course.lessonsList[currentIndex + 1]);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  return (
    <div className="flex flex-col h-full animate-in fade-in duration-500">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-8">
        <div className="flex items-center gap-4">
          <button 
            onClick={onBack}
            className="p-2 hover:bg-slate-800 rounded-lg text-slate-400 hover:text-white transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
          </button>
          <div>
            <h1 className="text-2xl font-bold text-white leading-tight">{course.title}</h1>
            <p className="text-slate-500 text-sm">Course Curriculum • {course.lessons} Lessons</p>
          </div>
        </div>
        
        <div className="bg-slate-900 border border-slate-800 rounded-2xl px-6 py-4 min-w-[200px] shadow-lg">
          <div className="flex justify-between items-center mb-2">
            <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Course Progress</span>
            <span className="text-[10px] font-bold text-emerald-400">{progressPercent}%</span>
          </div>
          <div className="w-full h-1.5 bg-slate-950 rounded-full overflow-hidden">
            <div 
              className="h-full bg-emerald-500 transition-all duration-1000 ease-out"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8 flex-1">
        {/* Sidebar: Lessons List */}
        <div className="lg:col-span-1 bg-slate-900 border border-slate-800 rounded-2xl p-4 h-fit lg:sticky lg:top-24">
          <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4 px-2">Lessons</h3>
          <div className="space-y-1">
            {course.lessonsList?.map((lesson, idx) => {
              const completed = completedLessonIds.includes(lesson.id);
              return (
                <button
                  key={lesson.id}
                  onClick={() => setActiveLesson(lesson)}
                  className={`w-full text-left px-4 py-3 rounded-xl text-sm transition-all flex items-center justify-between group ${
                    activeLesson?.id === lesson.id
                      ? 'bg-blue-600 text-white font-medium shadow-lg shadow-blue-600/20'
                      : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                  }`}
                >
                  <div className="flex items-center gap-3 truncate">
                    <span className={`w-6 h-6 shrink-0 rounded-full flex items-center justify-center text-[10px] border ${
                      activeLesson?.id === lesson.id 
                        ? 'bg-white/20 border-white/40' 
                        : completed 
                          ? 'bg-emerald-500/20 border-emerald-500/40 text-emerald-400' 
                          : 'bg-slate-950 border-slate-700'
                    }`}>
                      {completed ? (
                        <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                      ) : idx + 1}
                    </span>
                    <span className="truncate">{lesson.title}</span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Main Content Area */}
        <div className="lg:col-span-3 space-y-6">
          <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 md:p-10 shadow-xl overflow-hidden">
            {activeLesson ? (
              <>
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
                  <h2 className="text-3xl md:text-4xl font-extrabold text-white tracking-tight">{activeLesson.title}</h2>
                  <div className="shrink-0">
                    <span className="px-4 py-1 bg-blue-500/10 text-blue-400 text-xs font-bold rounded-full border border-blue-500/20">
                      Lesson {currentIndex + 1} of {course.lessonsList?.length}
                    </span>
                  </div>
                </div>
                
                <div className="max-w-none">
                  {/* Lesson Content using our MarkdownRenderer */}
                  <MarkdownRenderer content={activeLesson.content} />

                  {/* AI Context & Deep Dive - Dynamically Generated */}
                  <div className="bg-slate-950 rounded-2xl border border-slate-800 p-6 md:p-8 mt-12 relative overflow-hidden group">
                    <div className="absolute top-0 right-0 p-4 opacity-5 pointer-events-none group-hover:opacity-10 transition-opacity">
                      <svg className="w-24 h-24" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M12 2L4.5 20.29l.71.71L12 18l6.79 3 .71-.71z" />
                      </svg>
                    </div>
                    
                    <div className="flex items-center justify-between mb-6">
                      <div className="flex items-center gap-3 text-blue-400">
                        <div className="p-2 bg-blue-500/10 rounded-lg">
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                        </div>
                        <h4 className="font-bold text-sm uppercase tracking-widest">Gemini Deep Dive</h4>
                      </div>
                      <div className="px-3 py-1 rounded-full bg-slate-900 text-[10px] text-slate-500 font-mono border border-slate-800">
                        temp: 0.1
                      </div>
                    </div>
                    
                    {isGenerating ? (
                      <div className="space-y-4 animate-pulse">
                        <div className="h-4 bg-slate-800 rounded w-3/4"></div>
                        <div className="h-4 bg-slate-800 rounded w-full"></div>
                        <div className="h-4 bg-slate-800 rounded w-5/6"></div>
                      </div>
                    ) : (
                      <MarkdownRenderer content={lessonExplanation} />
                    )}
                  </div>
                </div>

                <div className="mt-12 flex flex-col sm:flex-row justify-between items-center gap-4 border-t border-slate-800 pt-8">
                  <button 
                    onClick={() => onToggleCompletion(activeLesson.id)}
                    className={`w-full sm:w-auto px-8 py-3 rounded-xl font-bold text-sm transition-all flex items-center justify-center gap-2 ${
                      isLessonCompleted 
                        ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 hover:bg-emerald-500/20' 
                        : 'bg-slate-800 text-slate-300 hover:bg-slate-700 hover:text-white'
                    }`}
                  >
                    {isLessonCompleted ? (
                      <>
                        <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                        Completed
                      </>
                    ) : 'Mark as Complete'}
                  </button>
                  
                  {!isLastLesson && (
                    <button 
                      onClick={handleNext}
                      className="w-full sm:w-auto px-8 py-3 rounded-xl bg-blue-600 text-white hover:bg-blue-500 transition-all font-bold text-sm shadow-lg shadow-blue-600/25 flex items-center justify-center gap-2 group"
                    >
                      Next Lesson
                      <svg className="w-4 h-4 transform group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
                      </svg>
                    </button>
                  )}
                </div>
              </>
            ) : (
              <div className="text-center py-20 text-slate-500">
                Select a lesson from the sidebar to begin your journey.
              </div>
            )}
          </div>

          {/* Quick Tips Box - Contextual help */}
          <div className="bg-gradient-to-r from-blue-600/10 to-emerald-500/10 border border-blue-500/20 rounded-2xl p-6 flex items-start gap-5">
             <div className="bg-blue-600 p-3 rounded-xl text-white shadow-lg shadow-blue-600/20">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.674a1 1 0 00.922-.606l2.024-4.512a1 1 0 00-.922-1.382H13v-2.19a1 1 0 00-.894-.994l-2.024-.225A1 1 0 009 8.19V12H6.914a1 1 0 00-.922 1.382l2.024 4.512a1 1 0 00.922.606z" />
                </svg>
             </div>
             <div>
                <h4 className="text-white font-bold mb-1">Developer Tip</h4>
                <p className="text-slate-400 text-sm leading-relaxed">
                  Try out the code snippets above in the <strong>Interactive Lab</strong>. Seeing how subtle prompt changes affect output is the best way to master LLM control.
                </p>
             </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CourseView;

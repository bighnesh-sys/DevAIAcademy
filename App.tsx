
import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import CourseCard from './components/CourseCard';
import AITutor from './components/AITutor';
import CodePlayground from './components/CodePlayground';
import CourseView from './components/CourseView';
import { COURSES, CATEGORIES } from './constants';
import { Course } from './types';

const App: React.FC = () => {
  const [activeCategory, setActiveCategory] = useState('All');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCourse, setSelectedCourse] = useState<Course | null>(null);
  const [completedLessons, setCompletedLessons] = useState<Record<string, string[]>>({});

  // Load progress from local storage
  useEffect(() => {
    const savedProgress = localStorage.getItem('devai_progress');
    if (savedProgress) {
      try {
        setCompletedLessons(JSON.parse(savedProgress));
      } catch (e) {
        console.error("Failed to parse progress", e);
      }
    }
  }, []);

  // Save progress to local storage
  useEffect(() => {
    localStorage.setItem('devai_progress', JSON.stringify(completedLessons));
  }, [completedLessons]);

  const toggleLessonCompletion = (courseId: string, lessonId: string) => {
    setCompletedLessons(prev => {
      const courseCompleted = prev[courseId] || [];
      if (courseCompleted.includes(lessonId)) {
        return { ...prev, [courseId]: courseCompleted.filter(id => id !== lessonId) };
      } else {
        return { ...prev, [courseId]: [...courseCompleted, lessonId] };
      }
    });
  };

  const getCourseProgress = (courseId: string) => {
    const completed = completedLessons[courseId] || [];
    const course = COURSES.find(c => c.id === courseId);
    if (!course || course.lessons === 0) return 0;
    return Math.round((completed.length / course.lessons) * 100);
  };

  const filteredCourses = COURSES.filter(course => {
    const matchesCategory = activeCategory === 'All' || course.category === activeCategory;
    const matchesSearch = course.title.toLowerCase().includes(searchQuery.toLowerCase()) || 
                          course.description.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  const handleStartCourse = (course: Course) => {
    setSelectedCourse(course);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleBackToCatalog = () => {
    setSelectedCourse(null);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleNavigation = (target: 'home' | 'courses' | 'playground') => {
    setSelectedCourse(null);
    
    setTimeout(() => {
      if (target === 'home') {
        window.scrollTo({ top: 0, behavior: 'smooth' });
      } else {
        const element = document.getElementById(target);
        element?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }, 0);
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Background patterns */}
      <div className="fixed inset-0 z-[-1] overflow-hidden pointer-events-none">
        <div className="absolute top-0 right-0 w-[800px] h-[800px] bg-blue-600/10 rounded-full blur-[120px] -translate-y-1/2 translate-x-1/2"></div>
        <div className="absolute bottom-0 left-0 w-[600px] h-[600px] bg-purple-600/10 rounded-full blur-[100px] translate-y-1/2 -translate-x-1/2"></div>
      </div>

      <Header onSearch={setSearchQuery} onNavigate={handleNavigation} />

      <main className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 w-full">
        {selectedCourse ? (
          <CourseView 
            course={selectedCourse} 
            onBack={handleBackToCatalog} 
            completedLessonIds={completedLessons[selectedCourse.id] || []}
            onToggleCompletion={(lessonId) => toggleLessonCompletion(selectedCourse.id, lessonId)}
          />
        ) : (
          <>
            {/* Hero Section */}
            <section className="mb-20 text-center">
              <h1 className="text-5xl md:text-7xl font-extrabold text-white mb-6 tracking-tight animate-in slide-in-from-top duration-700">
                Master the <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">AI Stack</span>
              </h1>
              <p className="text-xl text-slate-400 max-w-3xl mx-auto mb-10 leading-relaxed">
                The comprehensive platform for developers to bridge the gap between traditional engineering and artificial intelligence.
              </p>
              <div className="flex flex-wrap justify-center gap-4">
                <button 
                  onClick={() => handleNavigation('courses')}
                  className="bg-blue-600 hover:bg-blue-500 text-white px-8 py-3 rounded-full font-bold transition-all transform hover:scale-105 shadow-lg shadow-blue-500/25"
                >
                  Browse Courses
                </button>
                <button 
                  onClick={() => handleNavigation('playground')}
                  className="bg-slate-800 hover:bg-slate-700 text-slate-200 px-8 py-3 rounded-full font-bold border border-slate-700 transition-all transform hover:scale-105"
                >
                  Explore Playground
                </button>
              </div>
            </section>

            {/* Playground Section */}
            <section id="playground" className="mb-24 scroll-mt-24">
              <div className="flex items-center gap-2 mb-8">
                <span className="w-8 h-8 rounded-lg bg-emerald-500/20 flex items-center justify-center text-emerald-400">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                  </svg>
                </span>
                <h2 className="text-2xl font-bold text-white">Interactive Labs</h2>
              </div>
              <CodePlayground />
            </section>

            {/* Paradigm Guide Section */}
            <section className="mb-24 bg-slate-900/40 border border-slate-800 rounded-3xl p-8 backdrop-blur-sm">
              <div className="flex items-center gap-3 mb-8">
                <div className="p-2 bg-purple-500/20 rounded-lg text-purple-400">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.674a1 1 0 00.922-.606l2.024-4.512a1 1 0 00-.922-1.382H13v-2.19a1 1 0 00-.894-.994l-2.024-.225A1 1 0 009 8.19V12H6.914a1 1 0 00-.922 1.382l2.024 4.512a1 1 0 00.922.606z" />
                  </svg>
                </div>
                <h2 className="text-2xl font-bold text-white tracking-tight">Focus Guide: Generative vs. Agentic AI</h2>
              </div>
              
              <div className="grid md:grid-cols-2 gap-8 lg:gap-16">
                <div className="bg-slate-950/50 p-6 rounded-2xl border-l-4 border-blue-500">
                  <h4 className="text-blue-400 font-bold mb-4 uppercase tracking-widest text-xs">Generative AI Focus</h4>
                  <p className="text-slate-500 text-sm mb-6">Courses centered on content creation, semantic reasoning, and training predictive models.</p>
                  <ul className="space-y-4 text-slate-300 text-sm">
                    <li className="flex gap-2">
                      <span className="text-blue-500 font-bold">-</span>
                      <span><strong>Prompt Engineering:</strong> Optimizing token output and zero-shot reasoning.</span>
                    </li>
                    <li className="flex gap-2">
                      <span className="text-blue-500 font-bold">-</span>
                      <span><strong>Fine-tuning LLMs:</strong> Modifying model weights for domain-specific knowledge.</span>
                    </li>
                    <li className="flex gap-2">
                      <span className="text-blue-500 font-bold">-</span>
                      <span><strong>Visual Recognition:</strong> Deep learning for image classification and generation.</span>
                    </li>
                    <li className="flex gap-2">
                      <span className="text-blue-500 font-bold">-</span>
                      <span><strong>Fundamentals:</strong> The math behind weights, biases, and attention mechanisms.</span>
                    </li>
                  </ul>
                </div>

                <div className="bg-slate-950/50 p-6 rounded-2xl border-l-4 border-emerald-500">
                  <h4 className="text-emerald-400 font-bold mb-4 uppercase tracking-widest text-xs">Agentic AI Focus</h4>
                  <p className="text-slate-500 text-sm mb-6">Courses focused on autonomy, tool-use, and systems that execute multi-step workflows.</p>
                  <ul className="space-y-4 text-slate-300 text-sm">
                    <li className="flex gap-2">
                      <span className="text-emerald-500 font-bold">-</span>
                      <span><strong>Building AI Agents:</strong> Implementing the ReAct loop (Reason + Act).</span>
                    </li>
                    <li className="flex gap-2">
                      <span className="text-emerald-500 font-bold">-</span>
                      <span><strong>LangChain Mastery:</strong> Designing complex toolchains and autonomous state.</span>
                    </li>
                    <li className="flex gap-2">
                      <span className="text-emerald-500 font-bold">-</span>
                      <span><span><strong>RAG Architecture:</strong> Connecting models to live data for informed action.</span></span>
                    </li>
                    <li className="flex gap-2">
                      <span className="text-emerald-500 font-bold">-</span>
                      <span><strong>Toolkits & Memory:</strong> Giving agents the "hands" to use APIs and the "memory" to stay persistent.</span>
                    </li>
                  </ul>
                </div>
              </div>
            </section>

            {/* Course Catalog */}
            <section id="courses" className="mb-20 scroll-mt-24">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-12">
                <div>
                  <h2 className="text-3xl font-bold text-white mb-2">Learning Paths</h2>
                  <p className="text-slate-500">Structured curriculums designed by industry experts.</p>
                </div>
                
                <div className="flex flex-wrap gap-2">
                  {CATEGORIES.map(cat => (
                    <button
                      key={cat}
                      onClick={() => setActiveCategory(cat)}
                      className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
                        activeCategory === cat 
                        ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20' 
                        : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                      }`}
                    >
                      {cat}
                    </button>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filteredCourses.length > 0 ? (
                  filteredCourses.map(course => (
                    <CourseCard 
                      key={course.id} 
                      course={course} 
                      onStart={handleStartCourse}
                      progress={getCourseProgress(course.id)}
                    />
                  ))
                ) : (
                  <div className="col-span-full py-20 text-center bg-slate-800/20 rounded-2xl border border-dashed border-slate-700">
                    <p className="text-slate-500">No courses found matching your criteria.</p>
                  </div>
                )}
              </div>
            </section>

            {/* Stats Section */}
            <section className="mb-20 grid grid-cols-2 lg:grid-cols-4 gap-8">
              {[
                { label: 'Learners', val: '50k+' },
                { label: 'Total Lessons', val: '1.2k' },
                { label: 'Success Rate', val: '98%' },
                { label: 'AI Projects', val: '15k' },
              ].map((stat, i) => (
                <div key={i} className="text-center p-6 bg-slate-800/30 rounded-2xl border border-slate-800/50 hover:bg-slate-800/50 transition-colors">
                  <div className="text-3xl font-bold text-white mb-1">{stat.val}</div>
                  <div className="text-sm text-slate-500 font-medium uppercase tracking-wider">{stat.label}</div>
                </div>
              ))}
            </section>
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-slate-950 border-t border-slate-800 py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-12 mb-12">
            <div className="col-span-1 md:col-span-2">
              <div className="flex items-center gap-2 mb-6">
                <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center font-bold text-white text-xl">D</div>
                <span className="text-xl font-bold tracking-tight text-white">DevAI Academy</span>
              </div>
              <p className="text-slate-500 max-w-sm text-sm">
                Empowering the next generation of software engineers with the skills needed to build, deploy, and scale intelligent applications.
              </p>
            </div>
            <div>
              <h4 className="text-white font-bold mb-6 text-sm uppercase tracking-widest">Platform</h4>
              <ul className="space-y-4 text-slate-500 text-sm">
                <li><button onClick={() => handleNavigation('courses')} className="hover:text-blue-400 transition-colors">Courses</button></li>
                <li><button onClick={() => handleNavigation('playground')} className="hover:text-blue-400 transition-colors">Playground</button></li>
                <li><a href="#" className="hover:text-blue-400 transition-colors">Certifications</a></li>
                <li><a href="#" className="hover:text-blue-400 transition-colors">Enterprise</a></li>
              </ul>
            </div>
            <div>
              <h4 className="text-white font-bold mb-6 text-sm uppercase tracking-widest">Community</h4>
              <ul className="space-y-4 text-slate-500 text-sm">
                <li><a href="#" className="hover:text-blue-400 transition-colors">Discord Server</a></li>
                <li><a href="#" className="hover:text-blue-400 transition-colors">GitHub</a></li>
                <li><a href="#" className="hover:text-blue-400 transition-colors">Newsletter</a></li>
                <li><a href="#" className="hover:text-blue-400 transition-colors">Events</a></li>
              </ul>
            </div>
          </div>
          <div className="pt-8 border-t border-slate-900 flex flex-col md:flex-row justify-between items-center gap-4 text-slate-600 text-[10px] uppercase tracking-wider font-bold">
            <p>© 2024 DevAI Academy. Built with Gemini.</p>
            <div className="flex gap-8">
              <a href="#" className="hover:text-slate-400">Privacy Policy</a>
              <a href="#" className="hover:text-slate-400">Terms of Service</a>
              <a href="#" className="hover:text-slate-400">Cookie Policy</a>
            </div>
          </div>
        </div>
      </footer>

      <AITutor />
    </div>
  );
};

export default App;

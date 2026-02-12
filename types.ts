
export interface Lesson {
  id: string;
  title: string;
  content: string;
  isCompleted?: boolean;
}

export interface Course {
  id: string;
  title: string;
  description: string;
  category: 'LLM' | 'Computer Vision' | 'MLOps' | 'Fundamentals';
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced';
  duration: string;
  lessons: number;
  lessonsList?: Lesson[];
}

export interface ChatMessage {
  role: 'user' | 'model';
  content: string;
  timestamp: number;
}

export interface CodeSnippet {
  language: string;
  code: string;
  description: string;
}

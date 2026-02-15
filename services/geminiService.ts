
import { GoogleGenAI, Type } from "@google/genai";

/* Always use process.env.API_KEY directly in the GoogleGenAI constructor */
const getAIClient = () => {
  return new GoogleGenAI({ apiKey: process.env.API_KEY });
};

export const generateExplanation = async (topic: string): Promise<string> => {
  const ai = getAIClient();
  try {
    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: `Explain the AI concept of "${topic}" specifically for a software developer. Use analogies and mention implementation details.`,
      config: {
        systemInstruction: "You are a senior AI research scientist teaching software engineers. Keep it technical but accessible.",
        /* Set temperature to 0.1 for high consistency with minimal creative variance */
        temperature: 0.1,
      },
    });
    /* The text property directly returns the string output */
    return response.text || "No explanation available.";
  } catch (error) {
    console.error("Gemini Error:", error);
    return "Error connecting to the AI brain. Please check your network.";
  }
};

export const chatWithTutor = async (history: { role: string, parts: { text: string }[] }[]): Promise<string> => {
  const ai = getAIClient();
  try {
    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: history,
      config: {
        systemInstruction: "You are DevAI Tutor. You help developers learn AI concepts. Always provide code snippets in Python or TypeScript where applicable.",
        /* Set temperature to 0.1 for consistent tutor responses */
        temperature: 0.1,
      },
    });
    /* The text property directly returns the string output */
    return response.text || "I'm not sure how to answer that.";
  } catch (error) {
    console.error("Chat Error:", error);
    return "The tutor is currently offline. Try again later.";
  }
};

export const generateCodeSnippet = async (task: string): Promise<{ code: string, explanation: string }> => {
  const ai = getAIClient();
  try {
    const response = await ai.models.generateContent({
      model: 'gemini-3-pro-preview',
      contents: `Generate a production-ready code snippet for: ${task}. Return JSON with "code" and "explanation" fields.`,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            code: { type: Type.STRING },
            explanation: { type: Type.STRING },
          },
          required: ["code", "explanation"],
        },
        /* Code generation remains highly deterministic at 0.1 */
        temperature: 0.1,
      },
    });
    /* Accessing response.text is the correct way to get the generated content string */
    return JSON.parse(response.text || '{"code": "// No code generated", "explanation": "No explanation available"}');
  } catch (error) {
    return { code: "// Error generating code", explanation: "Failed to fetch code snippet." };
  }
};

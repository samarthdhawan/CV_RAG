"""
Resume RAG Pipeline
Parses Word resumes, extracts sections, and answers questions using retrieval.

Installation:
pip install python-docx scikit-learn huggingface-hub pyyaml

Config file (config.yaml):
params:
  huggingface_token: hf_YOUR_TOKEN_HERE
  model_name: meta-llama/Llama-3.2-3B-Instruct

input:
  cv: path/to/resume.docx
"""

import re
import yaml
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
from pathlib import Path

try:
    from docx import Document
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from huggingface_hub import InferenceClient
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install python-docx scikit-learn huggingface-hub pyyaml")
    raise


@dataclass
class ResumeSection:
    """Represents a section of the resume"""
    title: str
    content: str
    start_idx: int
    end_idx: int


class ResumeParser:
    """Parses resume documents and extracts structured sections"""
    
    SECTION_PATTERNS = [
        r'^(summary|professional summary|profile|objective)$',
        r'^(experience|work experience|employment|work history)$',
        r'^(education|academic background)$',
        r'^(skills|technical skills|core competencies|additional|technical)$',
        r'^(projects|key projects)$',
        r'^(certifications|certificates|licenses|certifications?\s*&\s*training)$',
        r'^(awards|achievements|honors)$',
        r'^(publications|research)$',
        r'^(languages|language proficiency)$',
        r'^(interests|hobbies)$',
        r'^(references|referees)$',
        r'^(soft skills)$',
    ]
    
    def __init__(self):
        self.section_pattern = re.compile('|'.join(self.SECTION_PATTERNS), re.IGNORECASE)
    
    def parse_docx(self, file_path: str) -> str:
        """Extract text from Word document"""
        doc = Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    
    def parse_file(self, file_path: str) -> str:
        """Parse resume file (Word documents only)"""
        path = Path(file_path)
        ext = path.suffix.lower()
        
        if ext in ['.docx', '.doc']:
            return self.parse_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Only .docx/.doc files are supported.")
    
    def extract_sections(self, text: str) -> List[ResumeSection]:
        """Extract sections from resume text"""
        lines = text.split('\n')
        sections = []
        current_section = None
        current_content = []
        start_idx = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            if self.section_pattern.match(line_stripped):
                if current_section:
                    sections.append(ResumeSection(
                        title=current_section,
                        content='\n'.join(current_content).strip(),
                        start_idx=start_idx,
                        end_idx=i
                    ))
                
                current_section = line_stripped
                current_content = []
                start_idx = i
            elif current_section and line_stripped:
                current_content.append(line_stripped)
        
        if current_section:
            sections.append(ResumeSection(
                title=current_section,
                content='\n'.join(current_content).strip(),
                start_idx=start_idx,
                end_idx=len(lines)
            ))
        
        return sections


class ResumeRAG:
    """RAG pipeline for resume question answering"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize RAG pipeline"""
        self.parser = ResumeParser()
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        self.config = self._load_config(config_path)
        
        self.client = InferenceClient(
            token=self.config['params']['huggingface_token']
        )
        self.model_name = self.config['params']['model_name']
        self.resume_path = self.config['input']['cv']
        
        self.sections: List[ResumeSection] = []
        self.section_vectors: Optional[np.ndarray] = None
        self.full_text: str = ""
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file with environment variable support"""
        with open(config_path, 'r') as f:
            config_str = f.read()
        
        # Replace environment variable placeholders
        import os
        import re
        
        # Find all ${VAR} patterns and replace with environment variables
        def replace_env_var(match):
            var_name = match.group(1)
            return os.getenv(var_name, '')
        
        config_str = re.sub(r'\$\{([^}]+)\}', replace_env_var, config_str)
        
        config = yaml.safe_load(config_str)
        return config
    
    def load_resume(self, file_path: str):
        """Load and process resume"""
        print(f"Loading resume from {file_path}...")
        
        self.full_text = self.parser.parse_file(file_path)
        self.sections = self.parser.extract_sections(self.full_text)
        print(f"Extracted {len(self.sections)} sections")
        
        if not self.sections:
            self.sections = [ResumeSection(
                title="Full Resume",
                content=self.full_text,
                start_idx=0,
                end_idx=len(self.full_text)
            )]
        
        texts = [f"{s.title} {s.content}" for s in self.sections]
        self.section_vectors = self.vectorizer.fit_transform(texts)
        
        print("Resume loaded and indexed successfully!")
    
    def retrieve_relevant_sections(self, query: str, top_k: int = 3) -> List[ResumeSection]:
        """Retrieve most relevant sections for a query"""
        if self.section_vectors is None:
            raise ValueError("No resume loaded. Call load_resume() first.")
        
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.section_vectors)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [self.sections[i] for i in top_indices]
    
    def answer_question(self, question: str, top_k: int = 3) -> str:
        """Answer a question about the resume"""
        relevant_sections = self.retrieve_relevant_sections(question, top_k)
        
        context = "\n\n".join([
            f"Section: {section.title}\n{section.content}"
            for section in relevant_sections
        ])
        
        user_prompt = f"""Based on the following sections from a resume, answer the question.

Resume Sections:
{context}

Question: {question}

Provide a clear, concise answer based only on the information in the resume sections above. If the information is not available, say so."""
        
        messages = [{"role": "user", "content": user_prompt}]
        
        response = self.client.chat_completion(
            messages=messages,
            model=self.model_name,
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def get_summary(self) -> str:
        """Generate a summary of the entire resume"""
        user_prompt = f"""Provide a concise professional summary of this resume, highlighting:
- Key qualifications and experience
- Main skills
- Career focus
- Notable achievements

Resume:
{self.full_text[:4000]}

Provide a 3-4 sentence summary."""
        
        messages = [{"role": "user", "content": user_prompt}]
        
        response = self.client.chat_completion(
            messages=messages,
            model=self.model_name,
            max_tokens=300,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def list_sections(self) -> List[str]:
        """List all extracted sections"""
        return [section.title for section in self.sections]


if __name__ == "__main__":
    rag = ResumeRAG(config_path="config.yaml")
    rag.load_resume(rag.resume_path)
    
    print("\nExtracted Sections:")
    for section in rag.list_sections():
        print(f"  - {section}")
    
    print("\n" + "="*50)
    print("RESUME SUMMARY")
    print("="*50)
    print(rag.get_summary())
    
    questions = [
        "What programming languages does the candidate know?",
        "What is their most recent work experience?",
        "What degree do they have?",
        "What are their key achievements?"
    ]
    
    print("\n" + "="*50)
    print("QUESTION ANSWERING")
    print("="*50)
    
    for question in questions:
        print(f"\nQ: {question}")
        answer = rag.answer_question(question)
        print(f"A: {answer}")
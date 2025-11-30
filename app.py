"""
Gradio interface for Resume RAG Chatbot
Deploy on Hugging Face Spaces
"""

import gradio as gr
from main import ResumeRAG
import os

# Initialize RAG pipeline
print("Loading RAG pipeline...")
rag = ResumeRAG(config_path="config.yaml")
rag.load_resume(rag.resume_path)
print("RAG pipeline loaded successfully!")

# Get resume sections for display
sections = rag.list_sections()
sections_text = "\n".join([f"• {section}" for section in sections])

def chat_with_resume(message, history):
    """
    Handle chat interactions
    Args:
        message: User's question
        history: Chat history (list of [user_msg, bot_msg] pairs)
    Returns:
        Updated history
    """
    try:
        # Get answer from RAG
        answer = rag.answer_question(message, top_k=3)
        return answer
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

def get_summary():
    """Generate resume summary"""
    try:
        return rag.get_summary()
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Sample questions for quick access
sample_questions = [
    "What programming languages do you know?",
    "Tell me about your work experience at Optum",
    "What is your educational background?",
    "What are your key achievements?",
    "What projects have you worked on?",
    "What tools and technologies are you familiar with?",
    "What certifications do you have?",
    "Tell me about your experience with Machine Learning"
]

# Custom CSS for better UI
custom_css = """
#main-header {
    text-align: center;
    font-size: 2.5em;
    font-weight: bold;
    color: #1f77b4;
    margin-bottom: 10px;
}
#sub-header {
    text-align: center;
    font-size: 1.2em;
    color: #666;
    margin-bottom: 20px;
}
.sample-question-btn {
    margin: 5px;
}
#resume-sections {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #dee2e6;
}
.footer {
    text-align: center;
    margin-top: 20px;
    padding: 10px;
    color: #666;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    # Header
    gr.HTML('<div id="main-header">💼 Chat with Samarth\'s Resume</div>')
    gr.HTML('<div id="sub-header">Ask me anything about my experience, skills, and background!</div>')
    
    with gr.Row():
        with gr.Column(scale=2):
            # Main chat interface
            chatbot = gr.Chatbot(
                height=500,
                show_label=False,
                avatar_images=(None, "🤖"),
                bubble_full_width=False
            )
            
            msg = gr.Textbox(
                placeholder="Ask a question about my resume...",
                show_label=False,
                scale=4
            )
            
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear Chat", scale=1)
            
            # Summary section
            with gr.Accordion("📄 View Resume Summary", open=False):
                summary_btn = gr.Button("Generate Summary")
                summary_output = gr.Textbox(
                    label="Professional Summary",
                    lines=6,
                    show_copy_button=True
                )
        
        with gr.Column(scale=1):
            # Sidebar with sample questions and info
            gr.Markdown("### 💡 Sample Questions")
            gr.Markdown("Click any question to ask:")
            
            # Create buttons for sample questions
            sample_question_btns = []
            for question in sample_questions:
                btn = gr.Button(question, size="sm")
                sample_question_btns.append((btn, question))
            
            gr.Markdown("---")
            
            # Resume sections
            gr.Markdown("### 📋 Resume Sections")
            gr.Markdown(
                f'<div id="resume-sections">{sections_text}</div>',
                elem_id="resume-sections"
            )
            
            gr.Markdown("---")
            
            # Contact info
            gr.Markdown("""
            ### 🔗 Connect with Me
            📧 samarthdhawan55@gmail.com
            
            🔗 [LinkedIn Profile](https://www.linkedin.com/in/samarth-dhawan)
            
            📍 Dublin, Ireland
            """)
    
    # Footer
    gr.HTML("""
    <div class="footer">
        Built with Gradio 🎨 | Powered by HuggingFace 🤗
    </div>
    """)
    
    # Event handlers
    def user_message(message, history):
        """Add user message to chat"""
        return "", history + [[message, None]]
    
    def bot_response(history):
        """Generate bot response"""
        message = history[-1][0]
        response = chat_with_resume(message, history)
        history[-1][1] = response
        return history
    
    # Submit message
    msg.submit(
        user_message,
        [msg, chatbot],
        [msg, chatbot],
        queue=False
    ).then(
        bot_response,
        chatbot,
        chatbot
    )
    
    submit_btn.click(
        user_message,
        [msg, chatbot],
        [msg, chatbot],
        queue=False
    ).then(
        bot_response,
        chatbot,
        chatbot
    )
    
    # Clear chat
    clear_btn.click(lambda: None, None, chatbot, queue=False)
    
    # Sample question buttons
    for btn, question in sample_question_btns:
        btn.click(
            lambda q=question: (q, []),
            None,
            [msg, chatbot],
            queue=False
        ).then(
            user_message,
            [msg, chatbot],
            [msg, chatbot],
            queue=False
        ).then(
            bot_response,
            chatbot,
            chatbot
        )
    
    # Summary button
    summary_btn.click(
        get_summary,
        None,
        summary_output
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()

import gradio as gr

test_css = """
@import url('https://fonts.googleapis.com/css?family=Roboto:400,500,700&display=swap');

:root {
  --md-primary: #1976d2;
  --md-primary-dark: #1565c0;
  --md-secondary: #424242;
  --md-background: #f9f9f9;
  --md-surface: #ffffff;
  --md-text: #212121;
  --md-text-on-primary: #ffffff;
  --md-border-radius: 8px;
  --md-transition: 0.3s ease;
}

/* 临时只用一层背景图，确保能发起请求 */
html, body, .gradio-container {
  margin: 0;
  padding: 0;
  font-family: 'Roboto', sans-serif;
  color: var(--md-text);

  /* 单层背景 + cover */
  background: url("https://raw.githubusercontent.com/EugeneYilia/JiAnAI/master/assets/images/freemasonry.png") 
              no-repeat center center / cover !important;
  background-color: transparent !important;
}
"""

def greet(name):
    return "Hello, " + name

demo = gr.Blocks(css=test_css)
with demo:
    gr.Markdown("## Minimal CSS Test")
    name_input = gr.Textbox(label="Your name")
    greet_btn = gr.Button("Greet")
    output = gr.Textbox(label="Greeting")

    greet_btn.click(fn=greet, inputs=name_input, outputs=output)

demo.launch()

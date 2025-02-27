const API_KEY = 'API KEY';
const API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent';

window.addEventListener('load', () => {
    marked.setOptions({
        highlight: function(code, lang) {
            if (!lang) {
                return code;
            }
            
            const languageMap = {
                'py': 'python',
                'js': 'javascript',
                'java': 'java',
                'html': 'markup',
                'xml': 'markup',
                'c': 'c',
                'cpp': 'cpp',
                'css': 'css',
                'json': 'json',
                'bash': 'bash',
                'sql': 'sql'
            };

            const langId = languageMap[lang.toLowerCase()] || lang.toLowerCase();

            if (Prism.languages[langId]) {
                return Prism.highlight(code, Prism.languages[langId], langId);
            }
            return code;
        },
        langPrefix: 'language-'
    });

    document.getElementById('sendButton').addEventListener('click', sendPrompt);
});

async function sendPrompt() {
    const promptInput = document.getElementById('promptInput');
    const responseArea = document.getElementById('responseArea');
    const prompt = promptInput.value.trim();

    if (!prompt) {
        responseArea.innerHTML = '';
        return;
    }
    
    responseArea.innerHTML = `
        <div class="formatted-response">
            <div class="loading">Lava is thinking...</div>
        </div>
    `;

    try {
        const response = await fetch(`${API_URL}?key=${API_KEY}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                contents: [{
                    parts: [{
                        text: prompt
                    }]
                }]
            })
        });

        const data = await response.json();

        if (data.candidates && data.candidates[0].content) {
            const aiResponse = data.candidates[0].content.parts[0].text;
            const formattedResponse = marked.parse(aiResponse);
            responseArea.innerHTML = `<div class="formatted-response">${formattedResponse}</div>`;

            document.querySelectorAll('pre code').forEach((block) => {
                Prism.highlightElement(block);
            });

            document.querySelectorAll('pre').forEach(addCopyButton);
        } else {
            throw new Error('Lava overload');
        }
    } catch (error) {
        responseArea.innerHTML = `<div class="error-message">Error: ${error.message}</div>`;
    }
}

function addCopyButton(pre) {
    const button = document.createElement('button');
    button.className = 'copy-button';
    button.textContent = 'Copy';
    
    button.addEventListener('click', () => {
        const code = pre.querySelector('code');
        navigator.clipboard.writeText(code.textContent).then(() => {
            button.textContent = 'Copied!';
            setTimeout(() => {
                button.textContent = 'Copy';
            }, 2000);
        });
    });
    
    pre.style.position = 'relative';
    pre.appendChild(button);
}

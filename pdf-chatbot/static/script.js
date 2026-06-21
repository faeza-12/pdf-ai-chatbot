async function uploadPDF() {

    const fileInput = document.getElementById("pdfFile");

    if (!fileInput.files.length) {
        alert("Please choose a PDF first.");
        return;
    }

    const formData = new FormData();

    formData.append(
        "file",
        fileInput.files[0]
    );

    document.getElementById("uploadStatus").innerText =
        "Uploading PDF...";

    const response = await fetch(
        "/upload",
        {
            method: "POST",
            body: formData
        }
    );

    const data = await response.json();

    document.getElementById("uploadStatus").innerText =
        "✅ PDF uploaded successfully";
}



async function askQuestion() {

    const question =
        document.getElementById("question").value;

    if (!question.trim()) return;

    const chat =
        document.getElementById("chatMessages");

    // User message

    chat.innerHTML += `
        <div class="user-message">
            ${question}
        </div>
    `;

    // Loading bubble

    chat.innerHTML += `
        <div class="bot-message" id="loading">
            🤖 Thinking...
        </div>
    `;

    chat.scrollTop = chat.scrollHeight;

    try {

        const response = await fetch(
            "http://127.0.0.1:8000/ask",
            {
                method: "POST",
                headers: {
                    "Content-Type":
                    "application/json"
                },
                body: JSON.stringify({
                    question: question
                })
            }
        );

        const data = await response.json();

        document
            .getElementById("loading")
            .remove();

        chat.innerHTML += `
            <div class="bot-message">
                🤖 ${data.answer}
            </div>
        `;

    } catch (error) {

        document
            .getElementById("loading")
            .remove();

        chat.innerHTML += `
            <div class="bot-message">
                ❌ Error getting response.
            </div>
        `;
    }

    document.getElementById("question").value = "";

    chat.scrollTop = chat.scrollHeight;
}



// Enter key support

document.addEventListener(
    "DOMContentLoaded",
    () => {

        const input =
            document.getElementById("question");

        input.addEventListener(
            "keypress",
            function(event) {

                if (event.key === "Enter") {
                    askQuestion();
                }

            }
        );

    }
);
document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeImageBtn = document.getElementById('remove-image-btn');
    const resultContainer = document.getElementById('result-container');
    const predictionSpan = document.getElementById('prediction');
    const confidenceSpan = document.getElementById('confidence');
    const spinner = document.getElementById('spinner');

    // --- Event Listeners ---

    // Trigger file input click when the upload area is clicked
    uploadArea.addEventListener('click', () => fileInput.click());

    // Handle file selection
    fileInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            handleFile(file);
        }
    });

    // Drag and Drop functionality
    uploadArea.addEventListener('dragover', (event) => {
        event.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (event) => {
        event.preventDefault();
        uploadArea.classList.remove('drag-over');
        const file = event.dataTransfer.files[0];
        if (file) {
            handleFile(file);
        }
    });

    // Handle image removal
    removeImageBtn.addEventListener('click', () => {
        resetUI();
    });

    // --- Core Functions ---

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }

        // Show image preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreviewContainer.style.display = 'block';
            uploadArea.style.display = 'none';
        };
        reader.readAsDataURL(file);

        // Get prediction
        getPrediction(file);
    }

    async function getPrediction(file) {
        // Reset previous results and show spinner
        resultContainer.style.display = 'none';
        spinner.style.display = 'block';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict/mnist', {
                method: 'POST',
                body: formData,
            });

            spinner.style.display = 'none';

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Prediction failed');
            }

            const data = await response.json();
            displayResult(data);

        } catch (error) {
            spinner.style.display = 'none';
            alert(`Error: ${error.message}`);
            resetUI();
        }
    }

    function displayResult(data) {
        predictionSpan.textContent = data.prediction;
        confidenceSpan.textContent = `${(data.confidence * 100).toFixed(2)}%`;
        resultContainer.style.display = 'block';
    }

    function resetUI() {
        // Clear file input
        fileInput.value = '';

        // Hide preview and result, show upload area
        imagePreviewContainer.style.display = 'none';
        resultContainer.style.display = 'none';
        spinner.style.display = 'none';
        uploadArea.style.display = 'block';
        imagePreview.src = '#';
    }
});
